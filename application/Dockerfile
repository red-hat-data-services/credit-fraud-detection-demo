FROM python:3.10.4

WORKDIR /tmp

COPY . /tmp

RUN pip install -r requirements.txt

ENV GRADIO_SERVER_PORT=8080
ENV GRADIO_SERVER_NAME=0.0.0.0
# ENV MLFLOW_ROUTE=http://mlflow-server.mlflow.svc.cluster.local:8080

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["model_application.py"]