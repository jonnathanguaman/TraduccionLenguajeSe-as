import base64
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
# Inicializar Flask
app = Flask(__name__)

CORS(app)
# Inicializar Mediapipe
mp_hands = mp.solutions.hands

# Cargar el modelo
model = tf.keras.models.load_model("modelo_leguaje_ecuatoriano.h5")

# Cargar el LabelEncoder guardado
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("clases_label_encoder.npy", allow_pickle=True)

# Corregir el padding de base64
def add_base64_padding(base64_string):
    padding = len(base64_string) % 4
    if padding != 0:
        base64_string += "=" * (4 - padding)
    return base64_string


# Función para preprocesar los puntos clave de la mano
def preprocess_landmarks(landmarks):
    data = []
    for landmark in landmarks:
        data.extend([landmark.x, landmark.y, landmark.z])
    return np.array(data).reshape(1, -1)

# Inicializar Mediapipe Hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)


@app.route('/predict', methods=['POST'])
def predict():
    # Recibir imagen del frontend (en formato base64)
    image_data = request.json.get("image")
    
    # Eliminar el encabezado de base64
    if image_data.startswith('data:image/jpeg;base64,'):
        image_data = image_data.replace('data:image/jpeg;base64,', '')

    # Corregir el padding de base64
    image_data = add_base64_padding(image_data)

    # Convertir de base64 a un arreglo de bytes
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)

    # Decodificar la imagen
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Error al decodificar la imagen"}), 400

    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    prediction = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            input_data = preprocess_landmarks(hand_landmarks.landmark)
            # Realizar la predicción
            model_prediction = model.predict(input_data)
            predicted_class = np.argmax(model_prediction, axis=1)
            predicted_label = label_encoder.inverse_transform(predicted_class)
            prediction = predicted_label[0]

    # Retornar la predicción
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
