
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY fase-2/requirements.txt fase-2/requirements.txt
COPY fase-3/requirements.txt fase-3/requirements.txt


RUN python -m pip install --no-cache-dir -r fase-3/requirements.txt

COPY fase-2/house_prices/ fase-2/house_prices/
COPY fase-2/train.py fase-2/predict.py fase-2/
COPY fase-3/api.py fase-3/api.py

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--app-dir", "fase-3", "--host", "0.0.0.0", "--port", "8000"]
