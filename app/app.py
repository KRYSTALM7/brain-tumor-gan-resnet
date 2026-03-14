import os
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier.keras")
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (256, 256)

def predict(image: Image.Image):
    img = image.resize(IMG_SIZE)
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "🔴 Tumor Detected" if prob >= 0.5 else "🟢 No Tumor"
    return label, round(prob * 100, 2), round((1 - prob) * 100, 2)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Brain MRI"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Tumor Probability (%)"),
        gr.Number(label="No Tumor Probability (%)"),
    ],
    title="Brain Tumor Detection — GAN + ResNet50",
    description="Upload a brain MRI scan. Model: ResNet50 trained with GAN augmentation.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)