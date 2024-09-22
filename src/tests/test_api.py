import os
import requests
import time
from fastapi.testclient import TestClient
from src.api.app import app  # Импорт вашего FastAPI приложения
import logging

client = TestClient(app)  # Используем TestClient для тестирования FastAPI

# Функция для получения токена аутентификации
def get_access_token():
    response = client.post("/token", data={"username": "admin", "password": "admin"})
    assert response.status_code == 200
    token = response.json().get("access_token")
    return token

def test_predict():
    # Получение токена для авторизации
    token = get_access_token()
    headers = {"Authorization": f"Bearer {token}"}

    # URL FastAPI для предсказаний
    url = "/predict/"

    # Путь к тестовому изображению
    test_image_path = 'src/data/test_images/POC_0.jpg'

    # Проверяем, что тестовое изображение существует
    assert os.path.exists(test_image_path), f"Test image not found at {test_image_path}"

    # Данные для отправки запроса с полным описанием
    data = {
        'designation': "L'Amour aux temps du choléra",
        'description': (
            "Quand on est un télégraphiste dont l’opulence est loin d’être le pain quotidien et une jeune écolière d’une famille plutôt aisée, l’amour est sans doute le meilleur moyen pour se compliquer la vie. Mais l’amour, aux Caraïbes, a pour sœur aînée la déraison. Florentino, amoureux de Fermina, va connaître cet état second dont les symptômes - plaisirs subtils de l’attente et souffrances de l’éloignement - sont si proches d’une maladie mortelle. Mais lorsqu’il commença à attendre la réponse à sa première lettre, son anxiété se compliqua de diarrhées et de vomissements verts, il perdit le sens de l’orientation, souffrant d’évanouissements subits."
        )
    }

    # Задержка для того, чтобы убедиться, что сервер готов принимать запросы
    time.sleep(2)

    # Отправка запроса
    with open(test_image_path, 'rb') as img_file:
        files = {'file': img_file}
        response = client.post(url, headers=headers, data=data, files=files)

    # Проверка статуса ответа
    assert response.status_code == 200, f"Unexpected status code {response.status_code}"
    response_data = response.json()
    assert "predicted_class" in response_data, "No 'predicted_class' in response"
    assert "f1_score_after" in response_data, "No 'f1_score_after' in response"

    # Проверка данных предсказания
    logging.info(f"Predicted class: {response_data['predicted_class']}")
    logging.info(f"F1-score after prediction: {response_data['f1_score_after']}")