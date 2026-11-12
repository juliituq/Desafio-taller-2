import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils import resample
import numpy as np
from scipy.sparse import load_npz

def cargar_dataset(dataset_path, nombre_predictor, etiquetas_path):
    if dataset_path.endswith(".npz"):
        X = load_npz(dataset_path)
    elif dataset_path.endswith(".npy"):
        X = np.load(dataset_path)
    etiquetas = pd.read_csv(etiquetas_path)
    y = etiquetas[nombre_predictor]
    return X, y


def balancear_clases(X, y, random_state=42):
    conteos = y.value_counts()
    objetivo = conteos.max()
    indices_balanceados = []
    for clase, cantidad in conteos.items():
        indices_clase = np.where(y.values == clase)[0]
        indices_muestreados = resample(
            indices_clase,
            replace=(cantidad < objetivo),
            n_samples=objetivo,
            random_state=random_state
        )
        indices_balanceados.extend(indices_muestreados)
    indices_balanceados = np.array(indices_balanceados)
    np.random.RandomState(random_state).shuffle(indices_balanceados)
    X_balanceado = X[indices_balanceados]
    y_balanceado = y.iloc[indices_balanceados].reset_index(drop=True)

    return X_balanceado, y_balanceado

datasets_train = {
    "dataset_bow_unigram": "Data/Final/bow_result_unigram_train.npz",
    "bow_result_unigram_bigram_train": "Data/Final/bow_result_unigram_bigram_train.npz",
    "tfidf_result_unigram_train": "Data/Final/tfidf_result_unigram_train.npz",
    "tfidf_result_unigram_bigram_train": "Data/Final/tfidf_result_unigram_bigram_train.npz", 
    "train_w2v_cbow": "Data/Final/train_w2v_cbow.npy", 
    "train_w2v_skipgram": "Data/Final/train_w2v_skipgram.npy", 
}
etiquetas_dataset = "Data/Splits/train.csv"
columna_y = "rating"
balancear = True

for nombre, ruta in datasets_train.items():
    modelos = {
    "Regresión Logística": LogisticRegression(max_iter=10000),
    "SVM Lineal": LinearSVC(),
    }
    if nombre.startswith(("dataset_bow", "bow", "tfidf")):
        modelos["Naive_Bayes"] = MultinomialNB()
    X, y = cargar_dataset(ruta, columna_y, etiquetas_dataset)
    if balancear:
        X_bal, y_bal = balancear_clases(X,y)

    resultados = {}
    for nombre_modelo, modelo in modelos.items():
        slug = nombre_modelo.lower().replace(" ", "_")
        modelo.fit(X_bal, y_bal)
        nombre_archivo = f"{nombre}_{slug}.pkl"
        ruta_modelo = os.path.join("Data","Models",nombre_archivo)
        joblib.dump(modelo, ruta_modelo)
        print(f"Modelo guardado en: {ruta_modelo}")
