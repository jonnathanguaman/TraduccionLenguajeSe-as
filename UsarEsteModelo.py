import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Inicializar Mediapipe
mp_hands = mp.solutions.hands

# Cargar el modelo
model = tf.keras.models.load_model("modelo_leguaje_ecuatoriano.h5")

# Cargar el LabelEncoder guardado
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("clases_label_encoder.npy", allow_pickle=True)

# Función para preprocesar los puntos clave de la mano
def preprocess_landmarks(landmarks):
    # Extraer las coordenadas x, y, z de cada punto de la mano
    data = []
    for landmark in landmarks:
        data.extend([landmark.x, landmark.y, landmark.z])
    return np.array(data).reshape(1, -1)  # Redimensionar a un formato adecuado para el modelo

# Inicializar Mediapipe Hands
with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los puntos clave en la mano
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Preprocesar los puntos clave
                input_data = preprocess_landmarks(hand_landmarks.landmark)

                # Realizar la predicción
                prediction = model.predict(input_data)

                # Decodificar la clase predicha utilizando el LabelEncoder
                predicted_class = np.argmax(prediction, axis=1)  # Índice de la clase con mayor probabilidad
                predicted_label = label_encoder.inverse_transform(predicted_class)

                # Mostrar el resultado de la predicción en la imagen
                cv2.putText(frame, f'Predicción: {predicted_label[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Mostrar la imagen con los puntos de la mano y la predicción
        cv2.imshow('Hand Gesture Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
