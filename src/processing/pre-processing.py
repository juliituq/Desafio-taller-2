import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords

#lista de palabras comunes 
nltk.download("stopwords", quiet=True)

dataset = pd.read_csv("Data/Raw/dataset.csv")
cantidad_inicial = len(dataset)


palabras_comunes = set(stopwords.words("english"))


palabras_comunes.difference_update({"no", "not", "nor", "never"})

negaciones = []

for palabra in palabras_comunes:
    if palabra.endswith("n't"):
        negaciones.append(palabra)

palabras_comunes.difference_update(negaciones)


def limpiar_texto(texto):
    if pd.isna(texto):
        return ""

    #quitar emojis y pasar a minúsculas
    texto = emoji.replace_emoji(texto, replace=" ")
    texto = texto.lower()

    
    texto = texto.replace("’", "'")

    #quitar enlaces
    palabras = texto.split()
    palabras_limpias = []

    for palabra in palabras:
        es_enlace = palabra.startswith(
            ("http://", "https://", "www.")
        )

        if not es_enlace:
            palabras_limpias.append(palabra)

    texto_limpio = " ".join(palabras_limpias)

    #reemplazar números por espacios
    texto_sin_numeros = ""

    for caracter in texto_limpio:
        if caracter.isdigit():
            texto_sin_numeros += " "
        else:
            texto_sin_numeros += caracter

    #reemplazar puntuación por espacios   
    texto_sin_puntuacion = ""

    for caracter in texto_sin_numeros:
        if caracter in '.,;:!?()[]{}"-/\\':
            texto_sin_puntuacion += " "
        else:
            texto_sin_puntuacion += caracter

    
    palabras = texto_sin_puntuacion.split()
    palabras_finales = []

    for palabra in palabras:
        if palabra not in palabras_comunes and len(palabra) > 1:
            palabras_finales.append(palabra)

    
    return " ".join(palabras_finales)


dataset["texto_limpio"] = dataset["text"].apply(limpiar_texto)

#quitar reseñas que quedaron vacías
tiene_texto = dataset["texto_limpio"] != ""
dataset = dataset[tiene_texto]

#guardar una copia limpia
dataset.to_csv("Data/Preprocessed/dataset_limpio.csv", index=False)

cantidad_final = len(dataset)

print("Reseñas iniciales:", cantidad_inicial)
print("Reseñas conservadas:", cantidad_final)
print("Reseñas vacías eliminadas:", cantidad_inicial - cantidad_final)

print("\nEjemplos de textos limpios:")
print(dataset["texto_limpio"].head())