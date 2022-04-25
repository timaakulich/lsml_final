FROM python:3.10-slim
WORKDIR /code
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY backend backend
