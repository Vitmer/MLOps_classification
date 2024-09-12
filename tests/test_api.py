import requests

# Define the URL for the FastAPI endpoint
url = "http://127.0.0.1:8000/predict/"

# Sample data
files = {'file': open('sample_image.jpg', 'rb')}
data = {
    'designation': 'Sample Product Name',
    'description': 'This is a sample product description for testing.'
}

# Send a POST request to the FastAPI endpoint
response = requests.post(url, data=data, files=files)

# Print the response
print(response.json())