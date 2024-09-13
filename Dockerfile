# Use a base Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libglib2.0-0

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip  
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]