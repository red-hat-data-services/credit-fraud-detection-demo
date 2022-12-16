# Import the dependencies we need to run the code.
import os
import mlflow.pyfunc
import gradio as gr
import numpy as np

# Get a few environment variables. These are so we can:
# - get data from MLFlow
# - Set server name and port for Gradio
MLFLOW_ROUTE = os.getenv("MLFLOW_ROUTE")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT"))
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME")

# Connect to MLFlow using the route.
mlflow.set_tracking_uri(MLFLOW_ROUTE)

# Specify what model and version we want to load, and then load it.
model_name = "DNN-credit-card-fraud"
model_version = 1
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)


# Create a small function that runs predictions on the loaded MLFlow model.
def predict(distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order):
    return "Fraud" if model.predict(np.array([[distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order]], dtype=np.float64))[0][0] >=0.995 else "Not fraud"


# Create and launch a Gradio interface that uses the prediction function to predict an output based on the inputs. 
# We also set up a few example inputs to make it easier to try out the application.
demo = gr.Interface(
    fn=predict, 
    inputs=["number","number","number","number","number","number","number"], 
    outputs="textbox",
    examples=[
        [57.87785658389723,0.3111400080477545,1.9459399775518593,1.0,1.0,0.0,0.0],
        [10.664473716016785,1.5657690862016613,4.886520843107555,1.0,0.0,0.0,1.0]
        ],
    title="Predict Credit Card Fraud"
    )

demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)
