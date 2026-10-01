from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WINDOW = 555

modelos_numeros = joblib.load("modelo_numeros.pkl")
modelos_estrellas = joblib.load("modelo_estrellas.pkl")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/predict-next")
def predict_next():

    try:

        df = pd.read_csv("euromillions_limpio.csv")

        # Crear matriz binaria
        rows = []

        for _, row in df.iterrows():

            registro = {}

            for n in range(1, 51):
                registro[f"num_{n}"] = 0

            for e in range(1, 13):
                registro[f"star_{e}"] = 0

            for n in [
                row["n1"],
                row["n2"],
                row["n3"],
                row["n4"],
                row["n5"]
            ]:
                registro[f"num_{int(n)}"] = 1

            for e in [
                row["e1"],
                row["e2"]
            ]:
                registro[f"star_{int(e)}"] = 1

            rows.append(registro)

        bin_df = pd.DataFrame(rows)

        # Última ventana
        ultima_ventana = bin_df.iloc[-WINDOW:]

        fila = {}

        for n in range(1, 51):
            fila[f"freq_num_{n}"] = (
                ultima_ventana[f"num_{n}"].sum()
            )

        for e in range(1, 13):
            fila[f"freq_star_{e}"] = (
                ultima_ventana[f"star_{e}"].sum()
            )

        X_pred = pd.DataFrame([fila])

        # Probabilidades números
        probs_numeros = []

        for numero, modelo in modelos_numeros.items():

            prob = modelo.predict_proba(X_pred)[0][1]

            probs_numeros.append({
                "numero": numero,
                "probabilidad": round(float(prob), 3)
            })

        probs_numeros.sort(
            key=lambda x: x["probabilidad"],
            reverse=True
        )

        # Probabilidades estrellas
        probs_estrellas = []

        for estrella, modelo in modelos_estrellas.items():

            prob = modelo.predict_proba(X_pred)[0][1]

            probs_estrellas.append({
                "estrella": estrella,
                "probabilidad": round(float(prob), 3)
            })

        probs_estrellas.sort(
            key=lambda x: x["probabilidad"],
            reverse=True
        )

        # Predicción principal
        numeros = sorted(
            [x["numero"] for x in probs_numeros[:5]]
        )

        estrellas = sorted(
            [x["estrella"] for x in probs_estrellas[:2]]
        )

        return {
            "numeros": numeros,
            "estrellas": estrellas,
            "top_numeros": probs_numeros[:10],
            "top_estrellas": probs_estrellas[:5]
        }

    except Exception as e:

        return {
            "error": str(e)
        }