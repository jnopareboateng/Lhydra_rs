import json
import requests
import time
import socket
from requests.exceptions import RequestException

def get_local_ip():
    """Get the local IP address of the machine"""
    return "127.0.0.1"  # Always use localhost for testing

def test_server_connection(url, max_retries=5, retry_delay=2):
    """Test server connection with retries"""
    print(f"Attempting to connect to server at {url}")
    for i in range(max_retries):
        try:
            health_url = f"{url}/api/health"
            print(f"Checking health endpoint at: {health_url}")
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') == 'healthy':
                    print("Server health check passed:")
                    print(f"- Status: {health_data.get('status')}")
                    print(f"- Model loaded: {health_data.get('model_loaded')}")
                    print(f"- Encoders loaded: {health_data.get('encoders_loaded')}")
                    return True
            print(f"Health check failed with status code: {response.status_code}")
            
        except requests.exceptions.ConnectionError:
            print(f"Attempt {i+1}/{max_retries}: Connection refused - is the server running?")
        except Exception as e:
            print(f"Attempt {i+1}/{max_retries}: Error - {str(e)}")
        
        if i < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    return False

# Server configuration
HOST = "127.0.0.1"
PORT = 5000
base_url = f"http://{HOST}:{PORT}"
print(f"Connecting to recommendation server at {base_url}")

# Test server connection first
if not test_server_connection(base_url):
    print("\nERROR: Server connection failed!")
    print("Please ensure:")
    print("1. The Flask app is running (python app.py)")
    print("2. Port 5000 is not in use")
    print(f"3. Server is accessible at {base_url}")
    exit(1)

# Define the payload with correct field names
data = {
    'age': 20,
    'gender': 'M',
    'favourite_genres': ['classical', 'jazz'],  # Changed from 'genre' to 'favourite_genres'
    'favourite_music': ['The Preludes and Fugues, Op. 87', 'The Nutcracker, Op. 71'],
    'favourite_artists': ['Shostakovich', 'Tchaikovsky']  # Changed from 'artist_name' to 'favourite_artists'
}

try:
    # Send a POST request with JSON data
    print("\nSending recommendation request...")
    print(f"Request data: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{base_url}/api/recommendations", json=data, timeout=30)
    
    if response.status_code == 200:
        print("\nRecommendations received successfully:")
        recommendations = response.json()
        print(json.dumps(recommendations, indent=2))
    else:
        print(f"\nError: {response.status_code}")
        print(f"Response: {response.text}")
except RequestException as e:
    print(f"\nError connecting to server: {str(e)}")
    print("Please ensure the Flask application is running and accessible.")
