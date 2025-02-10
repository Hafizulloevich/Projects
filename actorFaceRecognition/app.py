import gradio as gr
import cv2
import numpy as np
import pywt
import json
import joblib
import os
from PIL import Image

with open("saved_model.pkl", "rb") as f:
    model = joblib.load(f)

with open("class_dictionary.json", "r") as f:
    class_dict = json.load(f)

id_to_actor = {v: k.replace("_", " ").title() for k, v in class_dict.items()}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def get_cropped_image_if_2_eyes_from_array(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    return None

def w2d(img, mode='db1', level=5):
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray) / 255.0
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H


def preprocess_image(pil_img):
    img = np.array(pil_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    roi_color = get_cropped_image_if_2_eyes_from_array(img_bgr)
    if roi_color is None:
        roi_color = img_bgr
    scalled_raw_img = cv2.resize(roi_color, (32, 32))
    img_har = w2d(roi_color, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    raw_flat = scalled_raw_img.reshape(32*32*3, 1)
    har_flat = scalled_img_har.reshape(32*32, 1)
    combined_img = np.vstack((raw_flat, har_flat)).reshape(1, -1)
    return combined_img

def predict_actor(image):
    try:
        features = preprocess_image(image)
        prediction = model.predict(features)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence_dict = {id_to_actor[i]: float(proba[i]) for i in range(len(proba))}
        else:
            confidence_dict = {id_to_actor[prediction]: 1.0}

        return id_to_actor[prediction], confidence_dict
    except Exception as e:
        return f"Error: {e}", {}


def clear_inputs():
    """Clear uploaded image and outputs"""
    return None, None, None

examples = []
if os.path.exists("test_images"):
    for file in os.listdir("test_images"):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            examples.append(os.path.join("test_images", file))

with gr.Blocks() as demo:
    gr.Markdown("# 🎭 Actor Face Recognition")
    gr.Markdown("Upload an image of an actor and the model will predict who it is.")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload an Image", image_mode="RGB")
            with gr.Row():
                predict_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear")
        with gr.Column():
            label = gr.Label(label="Predicted Actor")
            confidences = gr.Label(label="Confidence Scores")

    gr.Examples(
        examples=examples,
        inputs=input_img,
        label="Test Images"
    )
    predict_btn.click(fn=predict_actor, inputs=input_img, outputs=[label, confidences])
    clear_btn.click(fn=clear_inputs, inputs=None, outputs=[input_img, label, confidences])

if __name__ == "__main__":
    demo.launch()


