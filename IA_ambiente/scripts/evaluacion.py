import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir el modelo
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Crear ImageDataGenerators para entrenamiento y prueba
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    '../dataset_entrenamiento',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    '../dataset_prueba',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)


# Cargar el modelo
model = tf.keras.models.load_model('../modelos/modelo_seguridad.h5')

# Evaluar el modelo
loss, accuracy = model.evaluate(test_generator)
print(f'Precisión del modelo en datos de prueba: {accuracy * 100:.2f}%')
