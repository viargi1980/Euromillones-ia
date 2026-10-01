import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

WINDOW = 555

# ==========================================
# CARGAR CSV
# ==========================================

df = pd.read_csv("euromillions_limpio.csv")

df["fecha"] = pd.to_datetime(df["fecha"])

df = df.sort_values("fecha").reset_index(drop=True)

# ==========================================
# MATRIZ DE NÚMEROS
# ==========================================

rows = []

for _, row in df.iterrows():

    registro = {}

    for n in range(1, 51):
        registro[f"num_{n}"] = 0

    for e in range(1, 13):
        registro[f"star_{e}"] = 0

    numeros = [
        row["n1"],
        row["n2"],
        row["n3"],
        row["n4"],
        row["n5"]
    ]

    estrellas = [
        row["e1"],
        row["e2"]
    ]

    for n in numeros:
        registro[f"num_{int(n)}"] = 1

    for e in estrellas:
        registro[f"star_{int(e)}"] = 1

    rows.append(registro)

bin_df = pd.DataFrame(rows)

# ==========================================
# FEATURES CON VENTANA
# ==========================================

dataset = []

for i in range(WINDOW, len(bin_df)):

    ventana = bin_df.iloc[i - WINDOW:i]

    fila = {}

    for n in range(1, 51):

        fila[f"freq_num_{n}"] = ventana[f"num_{n}"].sum()

    for e in range(1, 13):

        fila[f"freq_star_{e}"] = ventana[f"star_{e}"].sum()

    fila["target_index"] = i

    dataset.append(fila)

features = pd.DataFrame(dataset)

# ==========================================
# X
# ==========================================

X = features.drop(columns=["target_index"])

# ==========================================
# TRAIN TEST
# ==========================================

split = int(len(features) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

print()
print("Filas entrenamiento:", len(X_train))
print("Filas test:", len(X_test))
print("Features:", len(X.columns))
print()

# ==========================================
# ENTRENAR NÚMEROS
# ==========================================

modelos_numeros = {}

print("Entrenando números...")

for numero in range(1, 51):

    y = []

    for idx in features["target_index"]:

        y.append(
            bin_df.iloc[idx][f"num_{numero}"]
        )

    y = pd.Series(y)

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    modelo = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    modelos_numeros[numero] = modelo

print("OK")

# ==========================================
# ENTRENAR ESTRELLAS
# ==========================================

modelos_estrellas = {}

print("Entrenando estrellas...")

for estrella in range(1, 13):

    y = []

    for idx in features["target_index"]:

        y.append(
            bin_df.iloc[idx][f"star_{estrella}"]
        )

    y = pd.Series(y)

    y_train = y.iloc[:split]

    modelo = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    modelos_estrellas[estrella] = modelo

print("OK")

# ==========================================
# GUARDAR
# ==========================================

joblib.dump(
    modelos_numeros,
    "modelo_numeros.pkl"
)

joblib.dump(
    modelos_estrellas,
    "modelo_estrellas.pkl"
)

print()
print("Modelos guardados.")
print()

# ==========================================
# PREDICCIÓN
# ==========================================

ultima_ventana = bin_df.iloc[-WINDOW:]

fila = {}

for n in range(1, 51):
    fila[f"freq_num_{n}"] = ultima_ventana[f"num_{n}"].sum()

for e in range(1, 13):
    fila[f"freq_star_{e}"] = ultima_ventana[f"star_{e}"].sum()

X_pred = pd.DataFrame([fila])

# ==========================================
# PROBABILIDADES NÚMEROS
# ==========================================

probs_numeros = []

for numero, modelo in modelos_numeros.items():

    p = modelo.predict_proba(X_pred)[0][1]

    probs_numeros.append(
        (numero, p)
    )

probs_numeros.sort(
    key=lambda x: x[1],
    reverse=True
)

# ==========================================
# PROBABILIDADES ESTRELLAS
# ==========================================

probs_estrellas = []

for estrella, modelo in modelos_estrellas.items():

    p = modelo.predict_proba(X_pred)[0][1]

    probs_estrellas.append(
        (estrella, p)
    )

probs_estrellas.sort(
    key=lambda x: x[1],
    reverse=True
)

numeros_pred = sorted(
    [x[0] for x in probs_numeros[:5]]
)

estrellas_pred = sorted(
    [x[0] for x in probs_estrellas[:2]]
)




print("===================================")
print("PREDICCIÓN")
print("===================================")

print("Números :", numeros_pred)
print("Estrellas:", estrellas_pred)

print()
print("Top 10 números más probables:")
print()

for numero, p in probs_numeros[:10]:
    print(
        f"{numero:02d} -> {p:.3f}"
    )
