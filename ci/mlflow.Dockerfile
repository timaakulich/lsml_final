FROM python:3-slim
LABEL maintainer="Tima Akulich"
ARG MLFLOW_VERSION=1.25.1

WORKDIR /mlflow/
RUN pip install --no-cache-dir mlflow==$MLFLOW_VERSION
RUN pip install psycopg2-binary
EXPOSE 5000

ENV BACKEND_URI sqlite:////mlflow/mlflow.db
ENV ARTIFACT_ROOT /mlflow/artifacts

CMD mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --host 0.0.0.0 --port 5000
