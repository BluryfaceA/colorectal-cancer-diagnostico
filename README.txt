En este repositorio se presenta el código utilizado para la elaboración del trabajo de evaluación de imágenes de células cancerígenas colorectal:

Título: Deep Learning para la detección de cáncer colorrectal y análisis de imágenes histopatológicas

Autores:

Albarrán Jara Carlos Fernando.
Montenegro Baca Zee Ricardo.
Rodriguez Preciado André Jhonel. 

Directores: Juan Pedro Santos Fernández

Departamento: Ingeniería de Sistemas 

Universidad: Universidad Nacional de Trujillo (UNT)

Convocatoria: Julio 2025


Características técnicas:
Nvidia GeForce RTX 2060 6GB

Intel Core i7-9759H 2.60GHz

RAM 16GB

SSD 1TB

Windows 10 sistema 64bits

Python 3.8

Docker (para contenerizar la aplicación)

Bibliotecas principales: PyTorch, OpenCV, NumPy, Matplotlib, Scikit-learn

Primer paso: Crear el entorno con Docker
Construcción del contenedor:

bash
docker build -t colorectal-cancer-app .

Ejecución del contenedor:

bash
docker run -it --rm --gpus all -v $(pwd):/app colorectal-cancer-app

