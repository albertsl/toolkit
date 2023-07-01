import json

n = 34

with open('CountryCodesES.json', 'rb') as json_file:
    data = json.load(json_file, encoding='utf-8')
    for country in data:
        if country['dial_code'] == str(n):
            print(country['name'])