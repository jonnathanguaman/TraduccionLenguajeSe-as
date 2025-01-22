# Modelo de Inteligencia Artificial para Traducir Lenguaje de Señas

Este proyecto tiene como objetivo crear un modelo de inteligencia artificial capaz de traducir el lenguaje de señas de Ecuador. 

## Características

- **Versión**: 1.0
- **Lenguaje de Señas**: Se enfoca en la traducción de las letras **a**, **b**, y **c** en el lenguaje de señas de Ecuador.
- **Adaptabilidad**: El modelo está basado en los **21 puntos clave de la mano**, lo que permite que se pueda adaptar fácilmente a otros países que utilicen un sistema similar de puntos clave en el lenguaje de señas.
  
## Requisitos

Este proyecto requiere Python 3.10.16 para su correcta ejecución.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/jonnathanguaman/TraduccionDeLeguajeDeSe-asEcuador.git

2. Ejecuta el comando **pip install -r requirements.txt** desde la raiz 

3. Busca esta linea **save_landmarks_to_csv(hand_landmarks.landmark, label="c")** en el archivo **deteccionManos** y cambia el **label** por la letra que quieras guardar 

4. Ejecuta el archivo con **python .\deteccionManos.py**

5. Una vez que ya este escaneada la letra en la ternimal preciona **CONTROL C**

6. Realiza los pasos **3, 4 y 5** con las letras que desees  

7. Ejecuta el archivo **modeloPredictivo** para entrenar el modelo

9. El modelo esta listo para ser usado, puedes usarlo en el **escritorio** o en la **web**

10. Para ejecutar en el escritorio ejecuta el archivo **UsarEsteModelo** 

11. Para ejecutar en la web ejecuta el archivo **app.py** y abre el index en cualquier navegador

**Descripción de la red neuronal**

**Arquitectura de la red neuronal:**
**Capa de entrada:** La primera capa es una capa densa (Dense) con 128 neuronas y una función de activación ReLU (Rectified Linear Unit). Esta capa toma las características de entrada (X) y las procesa.
**Capa oculta: La segunda capa también es densa con 64 neuronas y una activación ReLU. Esta capa realiza el procesamiento intermedio entre la entrada y la salida.
**Capa de salida:** La última capa es una capa densa con el número de neuronas igual al número de clases en la variable de salida (y). La función de activación utilizada aquí es softmax, que es adecuada para tareas de clasificación múltiple porque transforma las salidas en probabilidades, donde la clase con la mayor probabilidad es la predicción final.

**Función de pérdida:**
El modelo utiliza la función de pérdida **categorical_crossentropy**, que es adecuada para problemas de clasificación múltiple con codificación one-hot (como es el caso en este código).

**Optimización:**
El optimizador utilizado es **Adam**, que es un optimizador popular y eficiente para problemas de aprendizaje profundo.

**Tipo de red neuronal:**
**Red Neuronal Multicapa (MLP):** Esta red es una red neuronal totalmente conectada **(feedforward)**, en la que cada neurona de una capa está conectada a todas las neuronas de la siguiente capa. Es un tipo básico de red neuronal utilizada en tareas de clasificación
