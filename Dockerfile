FROM python:3.12-slim


RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


RUN test -f artifacts/models.joblib || python train_models.py

ENV PORT=8080

CMD gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180
