import cv2
import mediapipe as mp
import csv
mp_hands = mp.solutions.hands

# Función para guardar puntos clave
def save_landmarks_to_csv(landmarks, label, file_name="leguajeEcuatoriano.csv"):
    with open(file_name, mode='a', newline='') as f:
        writer = csv.writer(f)
        row = []
        for landmark in landmarks:
            row.extend([landmark.x, landmark.y, landmark.z])  # Coordenadas x, y, z
        row.append(label)  # Etiqueta del gesto
        writer.writerow(row)

# Inicializar Mediapipe
with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
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

                # Guardar datos con una etiqueta
                save_landmarks_to_csv(hand_landmarks.landmark, label="c")  # Cambia la etiqueta según el gesto

        cv2.imshow('Hand Gesture Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
