FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY house_prices ./house_prices
COPY train.py predict.py ./

ENTRYPOINT ["python"]
CMD ["train.py", "--help"]