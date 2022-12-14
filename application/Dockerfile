FROM python:3.10.4

RUN pip install -r requirements.txt

ENV GRADIO_SERVER_PORT=7860
ENV MLFLOW_ROUTE=http://mlflow-server.mlflow.svc.cluster.local:8080

EXPOSE 7860

ENTRYPOINT ["python"]
CMD ["model_application.py"]