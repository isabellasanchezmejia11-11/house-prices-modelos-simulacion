FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY house_prices ./house_prices
COPY train.py predict.py ./

ENTRYPOINT ["python"]
CMD ["train.py", "--help"]
