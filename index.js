// Obtener acceso a la cámara
const video = document.getElementById('video');
const constraints = {
    video: true
};

navigator.mediaDevices.getUserMedia(constraints)
    .then(function(stream) {
        video.srcObject = stream;
    }).catch(function(error) {
        console.error("Error al acceder a la cámara: ", error);
    });

// Función para capturar una imagen desde la cámara
function captureImage() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg'); // Convertir a formato base64
}

// Función para hacer la predicción
async function makePrediction() {
    const image = captureImage(); // Capturar imagen desde la cámara
    const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: image }) // Enviar imagen completa en base64
    });

    const data = await response.json();
    document.getElementById("result").textContent = "Predicción: " + data.prediction;
}

// Hacer la predicción cada 1 segundo
setInterval(makePrediction, 1000);