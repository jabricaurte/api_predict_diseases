from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras
import numpy as np

app = FastAPI()

# cargar el modelo
modelo = keras.models.load_model("model.h5")

class Bovino(BaseModel):
    edad: int
    raza: int
    inseminacion: int
    tipo: float
    corral: float
    aborto: float
    terneros: float
    laboratorio: float

@app.post("/predict")
async def predecir(bovino: Bovino):

    # Llevar los datos al formato necesitado por keras
    lista_data_peticion = [bovino.edad, bovino.raza, bovino.inseminacion, bovino.tipo, bovino.corral, bovino.aborto, bovino.terneros, bovino.laboratorio]
    lista_data_peticion = list(map( lambda x: float(x), lista_data_peticion ))
    data_array = np.array(lista_data_peticion)
    data_tensor = data_array.reshape(1,8)

    # predecir utilizando el modelo y los datos
    predicciones = modelo.predict(data_tensor)

    # Extraer el resultado de la prediccion
    resultado = predicciones[0][0]
    if resultado >0.65:
        return {"Tiene la enfermedad"}
    else:
        return {"No tiene la enfermedad"}