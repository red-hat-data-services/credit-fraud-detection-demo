import os
import mlflow.pyfunc
import gradio as gr
import numpy as np

MLFLOW_ROUTE = os.getenv("MLFLOW_ROUTE")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT"))
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME")

mlflow.set_tracking_uri(MLFLOW_ROUTE)

model_name = "DNN-credit-fraud"
model_version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)



def predict(distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order):
    return model.predict(np.array([[distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order]], dtype=np.float64))[0][0] >=0.995


demo = gr.Interface(fn=predict, inputs=["number","number","number","number","number","number","number"], outputs="boolean")

demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)
