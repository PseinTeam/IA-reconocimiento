import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'C:/Users/Diego Jara/Desktop/Pruebas IA imagenes/IA_uniformes/dataset_entrenamiento',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'C:/Users/Diego Jara/Desktop/Pruebas IA imagenes/IA_uniformes/dataset_prueba',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)


model = create_model()
model.fit(train_generator, epochs=50, validation_data=test_generator)
model.save('C:/Users/Diego Jara/Desktop/Pruebas IA imagenes/IA_uniformes/modelos/modelo_uniformes.h5')
