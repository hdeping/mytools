import json

filename = "inchikey_filter_database_1.json"
#inchikey_filter_database_2.json
#inchikey_filter_database_3.json
#inchikey_filter_database_4.json
#inchikey_filter_database_5.json
#inchikey_filter_database_6.json

fp = open(filename,'r')
data = json.load(fp)

i=0
for key in data:
    i += 1
    print(key)
    output = json.dumps(data[key],indent=4)
    print(output)
    if i == 10:
        break


