# 🧬 Análisis de Cáncer Colorrectal con Redes Neuronales

Este proyecto implementa un sistema de clasificación de imágenes histológicas para detectar y analizar cáncer colorrectal (CRC) usando modelos de deep learning. Utiliza Streamlit para su despliegue interactivo y TensorFlow para la construcción de modelos convolucionales.

---

## 🧠 Tecnologías utilizadas

- Python 3.10+
- TensorFlow / Keras
- OpenCV
- Streamlit
- Matplotlib & Seaborn
- Scikit-learn
- FPDF (reportes)
- PIL / Pillow (imágenes)
- Keras Tuner (optimización de hiperparámetros)

---

## 🖥️ Requisitos del sistema (entorno de desarrollo)

Este software fue desarrollado y probado en la siguiente máquina:

- 💻 Dispositivo: DESKTOP-P7P6BN9  
- 🧮 Procesador: AMD Ryzen 7 5800H with Radeon Graphics @ 3.20 GHz  
- 🧠 Memoria RAM: 16 GB (15.4 GB utilizable)  
- 🖥️ Sistema operativo: Windows 10 - 64 bits (procesador x64)  
- 💾 Almacenamiento disponible: 1.38 TB  
- ⛔ Pantalla: Sin soporte táctil

> ✅ Recomendado: mínimo 8 GB de RAM y una GPU compatible con CUDA para acelerar el entrenamiento.

---

## 📁 Estructura del proyecto
├── app/ # Código principal de la app Streamlit
├── data/ # Imágenes de entrada (entrenamiento/test)
├── models/ # Modelos entrenados guardados
├── tuning/ # Optimización de hiperparámetros
├── reports/ # Reportes PDF generados
├── generate_metrics.py # Script para generar métricas del modelo
├── dockerfile # Definición del contenedor Docker
├── modelos.py # Implementación de las arquitecturas CNN
├── requirements.txt # Lista de dependencias

---

## 📊 Dataset utilizado

- 🧬 NCT-CRC-HE-100K: Dataset público con 100,000 imágenes histológicas coloreadas con hematoxilina y eosina (H&E), clasificado en varias categorías de tejido (tumoral, estroma, linfocitos, etc.).

Más información: https://zenodo.org/record/1214456

---

## ⚙️ Instalación

1. Clonar este repositorio:

```bash
git clone https://github.com/Gusva26/colorectal-cancer-analysis.git
cd colorectal-cancer-analysis

2. Crear entorno virtual e instalar requerimientos:

bash
python -m venv venv
venv\Scripts\activate          # En Windows
pip install -r requirements.txt

🚀 Ejecutar la app
Desde la raíz del proyecto:

bash
Copiar
Editar
streamlit run app.py


🧾 Requerimientos de Python
makefile
Copiar
Editar
streamlit==1.22.0
tensorflow==2.18.0
numpy==1.26.4
opencv-python==4.6.0.66
matplotlib==3.6.2
seaborn==0.12.1
fpdf
pillow==9.3.0
scikit-learn==1.2.0
keras-tuner==1.1.3
