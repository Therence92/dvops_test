FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN pip install -r requirement.txt

CMD ["python","app.py"]