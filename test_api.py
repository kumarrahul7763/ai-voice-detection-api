
import requests

url = "http://127.0.0.1:8000/predict"

headers = {
    "Authorization": "Bearer my_secret_key_123",
    "Content-Type": "application/json"
}

data = {
    "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    "message": "test"
}

response = requests.post(url, json=data, headers=headers)

print("Status Code:", response.status_code)
print("Response:", response.json())
