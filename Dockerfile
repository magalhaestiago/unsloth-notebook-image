FROM python:3.10-slim

# Evita prompts interativos durante instalações
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências básicas
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY requirements.txt .
COPY . .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Exponha a porta do Jupyter
EXPOSE 8888

# Comando para rodar o Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.disable_check_xsrf=True"]
