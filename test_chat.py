import requests
import json

# Load the query from the JSON file
with open('query.json', 'r') as f:
    payload = json.load(f)

# Define the URL and headers
url = "http://localhost:8000/chat"
headers = {"Content-Type": "application/json"}

try:
    # Send the POST request
    response = requests.post(url, headers=headers, json=payload)
    
    # Check if the request was successful
    response.raise_for_status()
    
    # Print the response from the server
    print(response.json())

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    if e.response:
        print(f"Response Body: {e.response.text}")
