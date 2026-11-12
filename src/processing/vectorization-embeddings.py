import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec

def vectorizar(texto, modelo):
    palabras = str(texto).split()
    vectores = [modelo.wv[p] for p in palabras if p in modelo.wv]

    if len(vectores) == 0:
        return np.zeros(modelo.vector_size)

    return np.mean(vectores, axis=0)

train = pd.read_csv("Data/Splits/train.csv")
valid = pd.read_csv("Data/Splits/validacion.csv")
test = pd.read_csv("Data/Splits/test.csv")

sentences = train["texto_limpio"].fillna("").str.split().tolist()

w2v_skipgram = Word2Vec(
    sentences, vector_size=150, window=5, min_count=5,
    sg=1, negative=5, epochs=10, workers=4, seed=67
)

w2v_cbow = Word2Vec(
    sentences, vector_size=150, window=5, min_count=5,
    sg=0, negative=5, epochs=10, workers=4, seed=67
)

Path("Data/Final").mkdir(parents=True, exist_ok=True)

w2v_skipgram.save("Data/Final/word2vec_skipgram.model")
w2v_cbow.save("Data/Final/word2vec_cbow.model")

for nombre, dataset in [
    ("train", train),
    ("valid", valid),
    ("test", test)
]:
    textos = dataset["texto_limpio"].fillna("")

    for tipo, modelo in [
        ("skipgram", w2v_skipgram),
        ("cbow", w2v_cbow)
    ]:
        vectores = np.array([
            vectorizar(texto, modelo) for texto in textos
        ])

        np.save(f"Data/Final/{nombre}_w2v_{tipo}.npy", vectores)

        print(f"{nombre} - {tipo}: {vectores.shape}")