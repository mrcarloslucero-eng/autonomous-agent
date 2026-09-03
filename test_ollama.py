import requests

try:
    response = requests.get("http://127.0.0.1:11434")
    print(f"Success! Status: {response.status_code}, Body: {response.text}")
except requests.exceptions.ConnectionError as e:
    print(f"Connection failed: {e}")