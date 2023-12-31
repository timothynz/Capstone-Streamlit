FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENV WATCHDOG_USE_POLLING=True

CMD streamlit run --server.port 8080 --server.enableCORS false app.py

