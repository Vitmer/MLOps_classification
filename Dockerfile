# Используйте базовый образ Amazon Linux 2
FROM amazonlinux:2

# Установите системные зависимости
RUN yum update -y && \
    yum install -y \
    python3 \
    python3-pip \
    gcc \
    gcc-c++ \
    libatomic \
    libstdc++-devel \
    && yum clean all

# Установите зависимости вашего приложения
COPY requirements.txt .

# Установите Python зависимости
RUN pip3 install --no-cache-dir -r requirements.txt

# Скопируйте приложение в контейнер
COPY . /app

# Установите рабочую директорию
WORKDIR /app

# Открывайте порты, если необходимо (например, для FastAPI)
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]