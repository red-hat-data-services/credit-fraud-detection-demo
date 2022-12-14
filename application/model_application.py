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
    return "Fraud" if model.predict(np.array([[distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order]], dtype=np.float64))[0][0] >=0.995 else "Not fraud"


demo = gr.Interface(
    fn=predict, 
    inputs=["number","number","number","number","number","number","number"], 
    outputs="textbox",
    examples=[
        [57.87785658389723,0.3111400080477545,1.9459399775518593,1.0,1.0,0.0,0.0],
        [10.664473716016785,1.5657690862016613,4.886520843107555,1.0,0.0,0.0,1.0]
        ],
    title="Predict Credit Fraud"
    )

demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)
