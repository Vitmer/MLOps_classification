import requests

# URL вашего запущенного API
url = 'http://35.158.255.92:8000/predict/'

# Путь к тестовому изображению
test_image_path = 'src/test_images/POC_0.jpg'  # Убедитесь, что путь правильный

# Данные для product designation и description
data = {
    'designation': 'L\'Amour aux temps du choléra',
    'description': 'Quand on est un télégraphiste dont l’opulence est loin d’être le pain quotidien et une jeune écolière d’une famille plutôt aisée, l’amour est sans doute le meilleur moyen pour se compliquer la vie. Mais l’amour, aux Caraïbes, a pour soeur aînée la déraison. Florentino, amoureux de Fermina, va connaître cet état second dont les symptômes - plaisirs subtils de l’attente et souffrances de l’éloignement - sont si proches d’une maladie mortelle. Mais lorsqu’il commença à attendre la réponse à sa première lettre, son anxiété se compliqua de diarrhées et de vomissements verts, il perdit le sens de l’orientation, souffrant d’évanouissements subits.'
}

# Отправка запроса с изображением и данными
with open(test_image_path, 'rb') as image_file:
    files = {'file': image_file}
    response = requests.post(url, data=data, files=files)

# Печать ответа от сервера
print(response.json())