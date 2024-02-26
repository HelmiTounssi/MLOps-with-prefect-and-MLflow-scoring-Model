FROM python:3.9

COPY  requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --ignore-installed -r requirements.txt
RUN pip install explainerdashboard
RUN pip install seaborn
RUN pip install phik
RUN pip install prettytable
RUN pip install bayesian-optimization
RUN pip install google-cloud-storage
RUN pip install gcsfs
COPY ["generate_dashboard.py", "run_dashboard.py", "./"]


# Helps us to know how to load the trained model
ENV IN_A_DOCKER_CONTAINER=True

# Create conda env based on a yaml file
RUN python generate_dashboard.py

EXPOSE 9050
CMD ["python", "./run_dashboard.py"]