import json

l = []
with open('CountryCodesES.json', 'rb') as json_file:
    data = json.load(json_file, encoding='utf-8')
    for country in data:
        l.append(int(country['dial_code']))

l.sort()

new_json = []
with open('CountryCodesES.json', 'rb') as json_file:
    data = json.load(json_file, encoding='utf-8')
    for i in l:
        for country in data:
            if int(country['dial_code']) == i:
                new_json.append(country)

with open('CountryCodesES2.json', 'w') as outfile:
    json.dump(new_json, outfile)