import requests

service_url = 'http://127.0.0.1:9696/predict' 

image_path = 'maine-coon.jpg'

with open(image_path, "rb") as file:
    imageData = file.read()

    files = {'file': ('image.jpg', imageData)}

    response = requests.post(service_url, files=files)


if response.status_code == 200:
    predictions = response.json()

    print(predictions)
else:
    print('Error:', response.content)
