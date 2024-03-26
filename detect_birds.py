import os
import cv2
import pandas as pd
from ultralytics import YOLO
import torch
import tensorflow as tf
# Clearing the memory before use
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.reset_memory_stats('GPU:0')
torch.cuda.empty_cache()
# Clearing the memory before use
mapping = {27:0, 37:1, 55:2, 58:3, 88:4,93:5, 97:6, 111:7, 120:8, 130:9, 138:10, 149:11}

model = YOLO("yolov8m.pt")
def load_and_return(model_path):
    return tf.keras.models.load_model(model_path)


def get_input_shape(model):
    input_shape = model.layers[0].input_shape
    input_shape = input_shape[0]
    if input_shape[1] == 224 and input_shape[2] == 224:
        return 224
    elif input_shape[1] == 299 and input_shape[2] == 299:
        return 299


def preprocess_name(name):
    name = name.lower()
    if ' ' in name:
        name = name.replace(' ', '_')
    if '-' in name:
        name = name.replace('-', '_')
    return name


def detect_birds(image_path):
    prediction = model.predict(image_path)
    result = prediction[0]
    birds = []
    for box in result.boxes:
        bounding_box = box.xyxy[0].tolist()
        image = cv2.imread(image_path)
        x_min, y_min, x_max, y_max = map(int, bounding_box)
        birds.append(image[y_min:y_max, x_min:x_max])
    return birds


def predict(image_path):
    tf.config.experimental.reset_memory_stats('GPU:0')
    detected_birds = detect_birds(image_path)
    bird_data = pd.read_csv("Species Family List and Status.csv")
    labels_167 = sorted(os.listdir("BIRDS1_split/train"))
    labels_165 = sorted(os.listdir("BIRDS1_split/train_165"))
    for idx, bird in enumerate(detected_birds):
        results = {}
        for model_name in os.listdir("Best_models"):
            model_path = os.path.join("Best_models", model_name)
            model = load_and_return(model_path)
            req_image_shape = get_input_shape(model)

            processed_bird_img = cv2.resize(bird, (req_image_shape, req_image_shape))
            prediction = model.predict(tf.expand_dims(processed_bird_img, axis=0)).argmax()
            if model_name == "best_vgg19_hdf5" or model_name == "best_eff_net_b1_hdf5":
                pred = labels_165[prediction - 1]
            else:
                if prediction in mapping:
                    prediction = mapping[prediction]
                pred = labels_167[prediction]
            if pred in results:
                results[pred] += 1
            else:
                results[pred] = 0
            tf.config.experimental.reset_memory_stats('GPU:0')

        result = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        result = preprocess_name(next(iter(result)))
        print(f"Bird number {idx} : ", bird_data[bird_data["Bird Species"] == result].to_dict(orient='records'))


predict("test_images/cropped_900 (1).jpg")
