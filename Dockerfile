
FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD streamlit run --server.port 8080 --server.enableCORS false --global.development.watch=False app.py

