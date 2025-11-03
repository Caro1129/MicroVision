# Usa una imagen base que ya tiene soporte para ciencia de datos
FROM python:3.11-slim

# Instala librerías del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1

# Establece el directorio de trabajo
WORKDIR /app

# Copia los requirements.txt e instálalos primero
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de tu código
COPY . .

# Comando para correr Streamlit
CMD ["streamlit", "run", "Microvision.py"]