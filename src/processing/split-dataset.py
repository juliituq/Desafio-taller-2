from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split



ROOT = Path(__file__).resolve().parents[2]

dataset = pd.read_csv(
    ROOT / "Data/Preprocessed/dataset_limpio.csv"
)

# Reservar el 20 % para test
resto, test = train_test_split(
    dataset,
    test_size=0.20,
    random_state=42,
    stratify=dataset["label"]
)

# El 25 % del 80 % restante equivale al 20 % del total
train, validacion = train_test_split(
    resto,
    test_size=0.25,
    random_state=42,
    stratify=resto["label"]
)

carpeta = ROOT / "Data/Splits"
carpeta.mkdir(parents=True, exist_ok=True)

train.to_csv(carpeta / "train.csv", index=False)
validacion.to_csv(carpeta / "validacion.csv", index=False)
test.to_csv(carpeta / "test.csv", index=False)

for nombre, datos in [
    ("Entrenamiento", train),
    ("Validación", validacion),
    ("Test", test)
]:
    print(f"\n{nombre}: {len(datos)} reseñas")
    print(datos["label"].value_counts().sort_index())