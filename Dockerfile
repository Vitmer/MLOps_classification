# Use a base image
FROM ubuntu:20.04

# Set non-interactive mode to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Install Python 3.10 and pip
RUN apt-get install -y python3.10 python3.10-distutils python3-pip

RUN pip install --upgrade pip

# Set Python 3.10 as the default Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10

# Upgrade pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Install Python development headers and necessary build tools
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    python3-pip \
    gcc \
    libhdf5-dev \
    pkg-config

# Set the working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Ensure the volumes for data and models are mounted
VOLUME /app/data
VOLUME /app/models

# Expose port 8000
EXPOSE 8000

# Run the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]