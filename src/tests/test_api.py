import requests

# Define the URL for the FastAPI endpoint
url = "http://127.0.0.1:8000/predict/"

# Path to the test image 'POC_0.jpg'
test_image_path = 'src/test_images/POC_0.jpg'  # Update path to the new location

# Sample data for product designation and description
data = {
    'designation': 'Test Product',
    'description': 'This is a test description for the product.'
}

# Open the test image and send it along with the form data to the server
with open(test_image_path, 'rb') as img_file:
    files = {'file': img_file}
    response = requests.post(url, data=data, files=files)

# Check if the request was successful
if response.status_code == 200:
    try:
        print(response.json())  # Print the prediction result
    except requests.exceptions.JSONDecodeError:
        print("Error: Could not decode the JSON response")
        print("Response text:", response.text)
else:
    print(f"Error: HTTP status code {response.status_code}")
    print("Response text:", response.text)  # Print the response text if something went wrong