# generate_metrics.py
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_specificity(cm):
    """Calcula la especificidad promedio"""
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    fp = cm.sum(axis=0) - np.diag(cm)
    return np.mean(tn / (tn + fp + 1e-9))

def process_existing_models():
    """Procesa modelos existentes y genera métricas"""
    results = {}
    
    # Mapeo de modelos (ajusta según tus archivos reales)
    model_mapping = {
        'CNN Simple': 'models/best_model.h5',
        'ResNet50': 'models/resnet_model.h5',
        'MobileNetV2': 'models/mobilenet_model.h5'
    }
    
    # Cargar dataset de validación (debes implementar esto)
    from modelos import val_gen  # Importa tu generador de validación
    
    for name, model_path in model_mapping.items():
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                val_gen.reset()
                y_true = val_gen.classes
                y_pred = np.argmax(model.predict(val_gen), axis=1)
                
                # Calcular métricas
                report = classification_report(y_true, y_pred, output_dict=True)
                cm = confusion_matrix(y_true, y_pred)
                
                # Guardar resultados
                results[name] = {
                    'classification_report': report,
                    'confusion_matrix': cm,
                    'mcc': matthews_corrcoef(y_true, y_pred),
                    'specificity': calculate_specificity(cm)
                }
                
                # Generar imagen de matriz de confusión
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Matriz de Confusión - {name}')
                plt.savefig(f'reports/{name}_confusion.png', bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error procesando {name}: {str(e)}")
    
    # Guardar datos para Streamlit
    os.makedirs('reports', exist_ok=True)
    with open('reports/evaluation_data.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    process_existing_models()