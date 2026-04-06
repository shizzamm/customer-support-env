FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]