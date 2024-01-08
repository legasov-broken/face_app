
FROM python:3.9-slim-buster
RUN apt-get update \
&& apt -y install software-properties-common  wget build-essential iputils-ping net-tools nano \
&& apt update && apt -y upgrade \
&& apt -y upgrade python3 libpython3-dev python3-oauth2client git python3-pip libopencv-dev python3-opencv curl wget libgl1-mesa-glx
# Use an official Python runtime as the base image


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt

# Copy the code into the container
COPY . .

# Expose the required port
EXPOSE 8080

# Run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:application
    
