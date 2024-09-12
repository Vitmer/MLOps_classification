import requests

# Define the URL for the FastAPI endpoint
url = "http://127.0.0.1:8000/predict/"

# Path to test image
test_image_path = 'src/test_images/POC_0.jpg'  # Change to the appropriate test image

# Sample data (product description and designation)
data = {
    'designation': 'Test Product Name',
    'description': 'This is a test product description for the FastAPI.'
}

# Open the test image and send it along with data to the server
with open(test_image_path, 'rb') as img_file:
    files = {'file': img_file}
    response = requests.post(url, data=data, files=files)

# Print the response (predicted class)
print(response.json())