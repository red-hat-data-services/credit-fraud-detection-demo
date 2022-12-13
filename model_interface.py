import mlflow.pyfunc
import gradio as gr

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_name = "DNN-credit-fraud"
model_version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)



def predict(distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order):
    return model.predict([[distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order]])[0][0]


demo = gr.Interface(fn=predict, inputs=["number","number","number","number","number","number","number"], outputs="number")

demo.launch()   
