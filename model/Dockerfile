FROM continuumio/miniconda3
RUN apt update
RUN apt install -y gcc

WORKDIR /app


COPY ["conda.yaml", "app_docker.py", "./"]

# Helps us to know how to load the trained model
ENV IN_A_DOCKER_CONTAINER=True

# Create conda env based on a yaml file
RUN conda env create --name mlflow-env --file conda.yaml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

EXPOSE 8090

# Make sure we use proper conda environment when running the gunicorn server
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "mlflow-env", "gunicorn", "--bind", "0.0.0.0:8090", "app_docker:server" ]

