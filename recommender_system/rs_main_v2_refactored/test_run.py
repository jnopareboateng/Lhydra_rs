import requests

# Define the URL of your API endpoint
url = "http://192.168.100.216:5000/api/recommendations"  # Use the server's IP address

# Define the payload with user data
data = {
    'user_id': 82418123798,
    'age': 33,
    'gender': 'male',
    'genre': 'highlife',
    'music': 'bells'
}

# Send a POST request with JSON data
response = requests.post(url, json=data)

# Print the response from the server
if response.status_code == 200:
    print("Recommendations received successfully:")
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)
