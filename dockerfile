# Imagen base ligera con Python y soporte para TensorFlow
FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos del entorno
COPY requirements.txt .
COPY app.py .
COPY modelos.py .
COPY models/ ./models/
COPY reports/ ./reports/
COPY NCT-CRC-HE-100K/ ./NCT-CRC-HE-100K/
COPY generate_metrics.py ./generate_metrics.py


# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Instalar dependencias del sistema para OpenCV y Matplotlib
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# Comando para ejecutar la app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
