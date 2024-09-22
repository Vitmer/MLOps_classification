# Используем Amazon Linux 2 как базовый образ
FROM amazonlinux:2

# Установим необходимые системные зависимости
RUN yum update -y && \
    yum install -y \
    python3 \
    python3-pip \
    gcc \
    gcc-c++ \
    libatomic \
    libstdc++-devel \
    hdf5 \
    hdf5-devel \
    pkgconfig \
    && yum clean all

# Обновим pip до последней версии
RUN pip3 install --upgrade pip

# Установим зависимости Python из файла requirements.txt
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Скопируем код приложения в контейнер
COPY . .

# Откроем порт для FastAPI
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]