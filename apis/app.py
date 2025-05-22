import requests

url = "http://localhost:8000/students/"

def put_data(name, age, fingers, is_a_bitch):
    data={
        'name' :name,
        'age':age,
        'extra':{
            'fingers' : fingers,
            'is_a_bitch':is_a_bitch
        }
    }
    response = requests.post(url, json=data)
    if response.status_code == 200 or response.status_code == 201:
        print("Successfully Inserted")
    else:
        print(f"Failed to post data: {response.status_code}")

    

params = {
    'offset': 0,
    'limit': 1   
}
response = requests.get(url,params=params)
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Failed to retrieve data: {response.status_code}")
