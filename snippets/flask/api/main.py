import requests

BASE = "http://127.0.0.1:5000/"

response = requests.get(BASE + "helloworld")
print(response.json())

response = requests.post(BASE + "helloworld")
print(response.json())

response = requests.get(BASE + "helloworld/Albert")
print(response.json())

# We can send additional data hidden in the POST request
response = requests.post(BASE + "helloworld_post", {"additional": "data", "as": "much", "data": "as", "we": "want"})
print(response.json())