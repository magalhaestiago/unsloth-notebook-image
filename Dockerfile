FROM jupyter/scipy-notebook:latest

USER root

COPY --chown=$NB_UID:$NB_GID requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt


USER $NB_UID
