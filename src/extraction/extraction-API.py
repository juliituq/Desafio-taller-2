import requests
import pandas as pd

URL = "https://amazon-reviews-api-g5ae.onrender.com/reviews"

def obtener_resenias(tamanio):
    lista_resenias = []
    offset = 0

    while True:
        parametros = {"limit": tamanio, "offset": offset}

        print("Solicitando reseñas desde:", offset)
        response = requests.get(URL, params=parametros, timeout=120)
        response.raise_for_status()

        respuesta = response.json()
        resenias = respuesta["data"]

        if not resenias:
            print("La API dejó de devolver reseñas antes de terminar.")
            return

        lista_resenias.extend(resenias)
        offset += respuesta["returned"]

        print("Reseñas descargadas:", len(lista_resenias))

        if offset >= respuesta["total_matching"]:
            break

    dataset = pd.DataFrame(lista_resenias)
    dataset.to_csv("Data/Raw/dataset.csv", index=False)

    print("Dataset guardado. Total de reseñas:", len(dataset))
    print(dataset.head())

obtener_resenias(tamanio=1000)
    