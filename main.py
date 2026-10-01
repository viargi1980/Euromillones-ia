import requests
import pandas as pd

url = "https://euromillions.api.pedromealha.dev/v1/draws"

r = requests.get(
    url,
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=30
)

data = r.json()

filas = []

for d in data:
    filas.append({
        "fecha": d["date"],
        "n1": d["numbers"][0],
        "n2": d["numbers"][1],
        "n3": d["numbers"][2],
        "n4": d["numbers"][3],
        "n5": d["numbers"][4],
        "e1": d["stars"][0],
        "e2": d["stars"][1]
    })

df = pd.DataFrame(filas)

df.to_csv("euromillions_limpio.csv", index=False)

print(df.head())