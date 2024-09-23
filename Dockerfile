# Use the Ubuntu 20.04 base image
FROM ubuntu:20.04

# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install dependencies
RUN apt-get update -y && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    gcc \
    libhdf5-dev \
    pkg-config \
    libpython3-dev

# Upgrade pip
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port and run the application
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]