# modelos_opt.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from time import time

# ===============================
# Configuración GPU
# ===============================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # usar memoria según necesidad
    except RuntimeError as e:
        print(e)

print("Dispositivos GPU disponibles:", gpus)

# ===============================
# Configuración global
# ===============================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'NCT-CRC-HE-100K'

# ===============================
# Cargar imágenes con flujo desde disco
# ===============================
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation'
)

# ===============================
# Modelos
# ===============================
def create_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_resnet_model(input_shape, num_classes):
    base_model = keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

def create_mobilenetv2_model(input_shape, num_classes):
    base_model = keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ===============================
# Entrenamiento y evaluación
# ===============================
models = {
    'CNN Simple': create_cnn_model((*IMG_SIZE, 3), train_gen.num_classes),
    'ResNet50': create_resnet_model((*IMG_SIZE, 3), train_gen.num_classes),
    'MobileNetV2': create_mobilenetv2_model((*IMG_SIZE, 3), train_gen.num_classes)
}

results = {}

for name, model in models.items():
    print(f"\n=== Entrenando modelo: {name} ===")
    start_time = time()
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )

    val_gen.reset()
    y_true = val_gen.classes
    y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    results[name] = {
        'model': model,
        'history': history,
        'test_accuracy': report['accuracy'] if 'accuracy' in report else report['weighted avg']['precision'],
        'classification_report': report,
        'confusion_matrix': cm,
        'training_time': time() - start_time
    }

# ===============================
# Guardar mejor modelo
# ===============================
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
best_model = results[best_model_name]['model']
print(f"\n✅ Mejor modelo: {best_model_name} con precisión: {results[best_model_name]['test_accuracy']:.4f}")

os.makedirs('models', exist_ok=True)
best_model.save(f'models/best_model.h5')

# ===============================
# Reportes visuales
# ===============================
def generate_reports(results, output_dir='reports'):
    os.makedirs(output_dir, exist_ok=True)
    
    for name, result in results.items():
        plt.figure(figsize=(12, 5))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(result['history'].history['accuracy'], label='Train')
        plt.plot(result['history'].history['val_accuracy'], label='Val')
        plt.title(f'{name} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Confusion matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{name.replace(" ", "_")}_report.png')
        plt.close()

generate_reports(results)
