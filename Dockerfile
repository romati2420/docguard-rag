# Imagen para Cloud Run (o cualquier runtime de contenedores). Cloud Run inyecta $PORT.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docguard/ docguard/
COPY data/ data/
COPY evals/ evals/

# Usuario sin privilegios: el proceso no corre como root.
RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

# GOOGLE_API_KEY se entrega como secreto en tiempo de ejecución (Secret Manager), nunca en la imagen.
# El índice FAISS se construye al primer arranque si no existe.
CMD ["sh", "-c", "uvicorn docguard.api:app --host 0.0.0.0 --port ${PORT}"]
