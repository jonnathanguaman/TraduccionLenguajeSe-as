import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import numpy as np

# Cargar los datos
data = pd.read_csv("leguajeEcuatoriano.csv")

# Separar características (X) y etiquetas (y)
X = data.iloc[:, :-1]  # Todas las columnas excepto la última
y = data.iloc[:, -1]   # Última columna (etiquetas)

# Convertir etiquetas a one-hot encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convertir etiquetas codificadas a formato one-hot
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_one_hot = one_hot_encoder.fit_transform(y_encoded.reshape(-1, 1))


# Definir la arquitectura del modelo
num_classes = y_one_hot.shape[1]  # Número de clases a predecir
input_shape = X.shape[1]  # Número de características

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_shape,)),  # Entrada con las características
    Dense(64, activation='relu'),  # Capa oculta
    Dense(num_classes, activation='softmax')  # Salida con tantas neuronas como clases
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X, y_one_hot, epochs=20, batch_size=32)

# Guardar el modelo en formato TensorFlow SavedModel
model.save("modelo_leguaje_ecuatoriano.h5")
# Guardar las clases del LabelEncoder
np.save("clases_label_encoder.npy", label_encoder.classes_)
