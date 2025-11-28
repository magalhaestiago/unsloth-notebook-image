from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import os
from contextlib import asynccontextmanager

# Global variables for model and tokenizer
model = None
tokenizer = None
is_training = False
training_status = {"status": "idle", "message": "No training in progress"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global model, tokenizer
    if os.path.exists("lora_model"):
        try:
            load_model_for_inference("lora_model")
            print("Loaded existing lora_model on startup")
        except Exception as e:
            print(f"Could not load existing model: {e}")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Unsloth Fine-tuning API",
    description="API for fine-tuning LLaMA models and generating responses",
    version="1.0.0",
    lifespan=lifespan
)




class TrainingConfig(BaseModel):
    model_name: str = "unsloth/Llama-3.2-3B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    output_dir: str = "outputs"
    save_model_path: str = "lora_model"


class DatasetRequest(BaseModel):
    """Request model for training with dataset."""
    model_name: Optional[str] = None  # Model name (e.g., "unsloth/Llama-3.2-3B-Instruct")
    dataset_name: Optional[str] = None  # HuggingFace dataset name
    dataset_data: Optional[List[List[Dict[str, str]]]] = None  # Direct dataset as list of conversations
    config: Optional[TrainingConfig] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class InferenceRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = 128
    temperature: float = 1.5
    min_p: float = 0.1


class InferenceResponse(BaseModel):
    response: str
    full_output: str


class ModelLoadRequest(BaseModel):
    model_path: str = "lora_model"


# ============ Helper Functions ============

def load_model_for_inference(model_path: str = "lora_model"):
    """Load a fine-tuned model for inference."""
    global model, tokenizer
    
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    
    FastLanguageModel.for_inference(model)
    return True


def run_training(
    dataset_name: Optional[str],
    dataset_data: Optional[List[List[Dict[str, str]]]],
    config: TrainingConfig
):
    """Run the fine-tuning process."""
    global model, tokenizer, is_training, training_status
    
    try:
        is_training = True
        training_status = {"status": "running", "message": "Initializing model..."}
        
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
        from datasets import load_dataset, Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments, DataCollatorForSeq2Seq
        
        # Load model
        training_status = {"status": "running", "message": f"Loading model {config.model_name}..."}
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=config.load_in_4bit,
        )
        
        # Apply LoRA
        training_status = {"status": "running", "message": "Applying LoRA adapters..."}
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Setup tokenizer with chat template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3.1",
        )
        
        # Formatting function for dataset
        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
            return {"text": texts}
        
        # Load dataset
        training_status = {"status": "running", "message": "Loading dataset..."}
        if dataset_name:
            dataset = load_dataset(dataset_name, split="train")
        elif dataset_data:
            # Convert list of conversations to Dataset
            dataset = Dataset.from_dict({"conversations": dataset_data})
        else:
            raise ValueError("Either dataset_name or dataset_data must be provided")
        
        # Apply formatting
        dataset = dataset.map(formatting_prompts_func, batched=True)
        
        # Setup trainer
        training_status = {"status": "running", "message": "Setting up trainer..."}
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                learning_rate=config.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=config.output_dir,
                report_to="none",
            ),
        )
        
        # Train on responses only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        
        # Train
        training_status = {"status": "running", "message": "Training in progress..."}
        trainer_stats = trainer.train()
        
        # Save model
        training_status = {"status": "running", "message": "Saving model..."}
        model.save_pretrained(config.save_model_path)
        tokenizer.save_pretrained(config.save_model_path)
        
        # Prepare model for inference
        FastLanguageModel.for_inference(model)
        
        training_time = round(trainer_stats.metrics['train_runtime'], 2)
        training_status = {
            "status": "completed",
            "message": f"Training completed in {training_time} seconds",
            "metrics": {
                "train_runtime": training_time,
                "train_loss": trainer_stats.metrics.get('train_loss', None),
            }
        }
        
    except Exception as e:
        training_status = {"status": "failed", "message": str(e)}
        raise e
    finally:
        is_training = False


# ============ API Endpoints ============

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Unsloth Fine-tuning API",
        "endpoints": {
            "POST /train": "Start fine-tuning with a dataset",
            "POST /inference": "Generate a response from the model",
            "GET /status": "Check training status",
            "POST /load-model": "Load a saved model for inference"
        }
    }


@app.get("/status")
async def get_status():
    """Get the current training status."""
    global is_training, training_status, model
    return {
        "is_training": is_training,
        "training_status": training_status,
        "model_loaded": model is not None
    }


@app.post("/train")
async def train_model(request: DatasetRequest, background_tasks: BackgroundTasks):
    """
    Start fine-tuning the model with the provided dataset.
    
    Parameters:
    - `model_name`: The base model to fine-tune (e.g., "unsloth/Llama-3.2-3B-Instruct")
    - `dataset_name`: A HuggingFace dataset name (e.g., "mlabonne/FineTome-100k")
    - `dataset_data`: A list of conversations directly
    
    Example request:
    ```json
    {
        "model_name": "unsloth/Llama-3.2-3B-Instruct",
        "dataset_name": "mlabonne/FineTome-100k"
    }
    ```
    
    Or with custom dataset:
    ```json
    {
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
        "dataset_data": [
            [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing great, thank you!"}
            ],
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        ]
    }
    ```
    """
    global is_training
    
    if is_training:
        raise HTTPException(status_code=409, detail="Training is already in progress")
    
    if not request.dataset_name and not request.dataset_data:
        raise HTTPException(
            status_code=400, 
            detail="Either dataset_name or dataset_data must be provided"
        )
    
    config = request.config or TrainingConfig()
    
    # Override model_name if provided directly in request
    if request.model_name:
        config.model_name = request.model_name
    
    # Run training in background
    background_tasks.add_task(
        run_training,
        request.dataset_name,
        request.dataset_data,
        config
    )
    
    return {
        "message": "Training started",
        "model_name": config.model_name,
        "config": config.model_dump(),
        "dataset_source": request.dataset_name or "provided data"
    }


@app.post("/inference", response_model=InferenceResponse)
async def generate_response(request: InferenceRequest):
    """
    Generate a response from the fine-tuned model.
    
    Example request:
    ```json
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_new_tokens": 128,
        "temperature": 1.5,
        "min_p": 0.1
    }
    ```
    """
    global model, tokenizer, is_training
    
    if is_training:
        raise HTTPException(status_code=409, detail="Model is currently training")
    
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=400, 
            detail="No model loaded. Please train a model first or load an existing one."
        )
    
    try:
        from unsloth import FastLanguageModel
        
        # Ensure model is in inference mode
        FastLanguageModel.for_inference(model)
        
        # Convert messages to the expected format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Tokenize input
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate response
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=request.max_new_tokens,
            use_cache=True,
            temperature=request.temperature,
            min_p=request.min_p
        )
        
        # Decode output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        response_text = tokenizer.decode(
            outputs[0][inputs.shape[1]:], 
            skip_special_tokens=True
        )
        
        return InferenceResponse(
            response=response_text.strip(),
            full_output=full_output
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-model")
async def load_model_endpoint(request: ModelLoadRequest):
    """Load a previously saved model for inference."""
    global is_training
    
    if is_training:
        raise HTTPException(status_code=409, detail="Cannot load model while training is in progress")
    
    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=404, detail=f"Model not found at {request.model_path}")
    
    try:
        load_model_for_inference(request.model_path)
        return {"message": f"Model loaded successfully from {request.model_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu-info")
async def get_gpu_info():
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA is not available"}
    
    gpu_stats = torch.cuda.get_device_properties(0)
    reserved_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    total_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    return {
        "gpu_name": gpu_stats.name,
        "total_memory_gb": total_memory,
        "reserved_memory_gb": reserved_memory,
        "memory_usage_percent": round(reserved_memory / total_memory * 100, 2)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)