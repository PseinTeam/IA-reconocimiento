import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Cambia esta ruta si el archivo del modelo está en una ubicación diferente
model_path = 'C:/Users/Diego Jara/Desktop/Pruebas IA imagenes/modelos/modelo_seguridad.h5'

# Cargar el modelo
try:
    model = tf.keras.models.load_model(model_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

def predict_image(img_path):
    # Cargar y procesar la imagen
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizar
    
    # Hacer la predicción
    predictions = model.predict(img_array)
    return predictions[0][0]

# Ruta a la imagen de prueba
img_path = 'C:/Users/Diego Jara/Desktop/Pruebas IA imagenes/dataset_prueba/zona_segura/imagen4.jpg'

# Realizar la predicción
resultado = predict_image(img_path)
print(f'Predicción: {"Zona segura" if resultado >= 0.5 else "Falla de seguridad"}')
