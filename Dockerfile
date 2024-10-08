FROM python:3.11.0-slim
WORKDIR /app
COPY . /app
RUN apt update -y && apt install awscli -y
RUN pip install -r requirements.txt
ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0" ]
