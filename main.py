import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import studentized_range

# Cargar el archivo CSV en el DataFrame
df = pd.read_csv('data/datos_estudio.csv')

# Crear un Diccionario para almacenar sumas
suma_columnas = {}

# Realizar el cálculo de la suma de cada columna
for columna in df.columns:
    suma_columnas[columna] = round(df[columna].sum(), 4)

# Calcular la suma total de todas las columnas
suma_total = round(sum(suma_columnas.values()), 4)

# Crear un Diccionario para almacenar las sumas de los valores al cuadrado
suma_cuadrados = {}

# Realizar el cálculo de la suma de los valores al cuadrado para cada columna
for columna in df.columns:
    suma_cuadrados[columna] = round((df[columna] ** 2).sum(), 4)

# Calcular la suma total de las sumas de los cuadrados
suma_total_cuadrados = round(sum(suma_cuadrados.values()), 4)

# Crear un Diccionario para almacenar la cantidad de filas por columna
cantidad_filas = {}

# Contar la cantidad de filas en cada columna
for columna in df.columns:
    cantidad_filas[columna] = len(df[columna])

# Crear el DataFrame con los valores originales y sus cuadrados
df_resultados = pd.DataFrame({
    "y (Oxido_nitroso)": df["Oxido_nitroso"].round(4),
    "y²": (df["Oxido_nitroso"] ** 2).round(4),
    "x1 (Humedad)": df["Humedad"].round(4),
    "x1²": (df["Humedad"] ** 2).round(4),
    "x2 (Temperatura)": df["Temperatura"].round(4),
    "x2²": (df["Temperatura"] ** 2).round(4),
    "x3 (Presion)": df["Presion"].round(4),
    "x3²": (df["Presion"] ** 2).round(4)
})

# Agregar la fila de sumas de Σxt
df_resultados.loc["Σxt"] = {
    "y (Oxido_nitroso)": suma_columnas["Oxido_nitroso"],
    "y²": "",
    "x1 (Humedad)": suma_columnas["Humedad"],
    "x1²": "",
    "x2 (Temperatura)": suma_columnas["Temperatura"],
    "x2²": "",
    "x3 (Presion)": suma_columnas["Presion"],
    "x3²": "",
}

# Añadir el total de las sumas en "Σxt"
df_resultados.at["Σxt", "Total suma"] = round(
    suma_columnas["Oxido_nitroso"] + suma_columnas["Humedad"] +
    suma_columnas["Temperatura"] + suma_columnas["Presion"], 4
)

# Agregar la fila de Σxt²
df_resultados.loc["Σxt²"] = {
    "y (Oxido_nitroso)": "",
    "y²": suma_cuadrados["Oxido_nitroso"],
    "x1 (Humedad)": "",
    "x1²": suma_cuadrados["Humedad"],
    "x2 (Temperatura)": "",
    "x2²": suma_cuadrados["Temperatura"],
    "x3 (Presion)": "",
    "x3²": suma_cuadrados["Presion"],
}

# Añadir el total de los cuadrados en "Σxt²"
df_resultados.at["Σxt²", "Total suma"] = round(
    suma_cuadrados["Oxido_nitroso"] + suma_cuadrados["Humedad"] +
    suma_cuadrados["Temperatura"] + suma_cuadrados["Presion"], 4
)

# Calcular el cuadrado de las sumas totales
suma_totales_cuadrado = {
    "y (Oxido_nitroso)": suma_columnas["Oxido_nitroso"] ** 2,
    "x1 (Humedad)": suma_columnas["Humedad"] ** 2,
    "x2 (Temperatura)": suma_columnas["Temperatura"] ** 2,
    "x3 (Presion)": suma_columnas["Presion"] ** 2
}

# Calcular la suma total de los cuadrados
total_suma_cuadrados = sum(suma_totales_cuadrado.values())

# Añadir la fila de los (Σxt)²
df_resultados.loc["(Σxt)²"] = {
    "y (Oxido_nitroso)": round(suma_totales_cuadrado["y (Oxido_nitroso)"], 4),
    "y²": "",
    "x1 (Humedad)": round(suma_totales_cuadrado["x1 (Humedad)"], 4),
    "x1²": "",
    "x2 (Temperatura)": round(suma_totales_cuadrado["x2 (Temperatura)"], 4),
    "x2²": "",
    "x3 (Presion)": round(suma_totales_cuadrado["x3 (Presion)"], 4),
    "x3²": "",
}

# Añadir el total de los (Σxt)²
df_resultados.at["(Σxt)²", "Total suma"] = round(total_suma_cuadrados, 4)

# Crear la fila 'n' con el número de elementos por cada columna
df_resultados.loc["nt"] = {
    "y (Oxido_nitroso)": cantidad_filas["Oxido_nitroso"],
    "y²": "",
    "x1 (Humedad)": cantidad_filas["Humedad"],
    "x1²": "",
    "x2 (Temperatura)": cantidad_filas["Temperatura"],
    "x2²": "",
    "x3 (Presion)": cantidad_filas["Presion"],
    "x3²": "",
}

# Añadir el total de las 'n' en "n"
total_nt = cantidad_filas["Oxido_nitroso"] + cantidad_filas["Humedad"] + cantidad_filas["Temperatura"] + cantidad_filas["Presion"]
df_resultados.at["nt", "Total suma"] = int(total_nt)  # Convierte el total a entero


# Crear la fila '(Σxt)² / nt' dividiendo cada suma total al cuadrado entre su respectivo 'nt'
df_resultados.loc["(Σxt)² / nt"] = {
    "y (Oxido_nitroso)": round(suma_totales_cuadrado["y (Oxido_nitroso)"] / cantidad_filas["Oxido_nitroso"], 4),
    "y²": "",
    "x1 (Humedad)": round(suma_totales_cuadrado["x1 (Humedad)"] / cantidad_filas["Humedad"], 4),
    "x1²": "",
    "x2 (Temperatura)": round(suma_totales_cuadrado["x2 (Temperatura)"] / cantidad_filas["Temperatura"], 4),
    "x2²": "",
    "x3 (Presion)": round(suma_totales_cuadrado["x3 (Presion)"] / cantidad_filas["Presion"], 4),
    "x3²": "",
}

# Calcular el total correcto de (Σxt)² / nt
total_xt2_n = (
    suma_totales_cuadrado["y (Oxido_nitroso)"] / cantidad_filas["Oxido_nitroso"] +
    suma_totales_cuadrado["x1 (Humedad)"] / cantidad_filas["Humedad"] +
    suma_totales_cuadrado["x2 (Temperatura)"] / cantidad_filas["Temperatura"] +
    suma_totales_cuadrado["x3 (Presion)"] / cantidad_filas["Presion"]
)

df_resultados.at["(Σxt)² / nt", "Total suma"] = round(total_xt2_n, 4)

# Calcular la media (x̅) de cada columna
df_resultados.loc["x̅ (Media)"] = {
    "y (Oxido_nitroso)": round(suma_columnas["Oxido_nitroso"] / cantidad_filas["Oxido_nitroso"], 4),
    "y²": "",
    "x1 (Humedad)": round(suma_columnas["Humedad"] / cantidad_filas["Humedad"], 4),
    "x1²": "",
    "x2 (Temperatura)": round(suma_columnas["Temperatura"] / cantidad_filas["Temperatura"], 4),
    "x2²": "",
    "x3 (Presion)": round(suma_columnas["Presion"] / cantidad_filas["Presion"], 4),
    "x3²": "",
}

# Calcular el total de la media sumando los valores de cada columna
total_media = (
    suma_columnas["Oxido_nitroso"] / cantidad_filas["Oxido_nitroso"] +
    suma_columnas["Humedad"] / cantidad_filas["Humedad"] +
    suma_columnas["Temperatura"] / cantidad_filas["Temperatura"] +
    suma_columnas["Presion"] / cantidad_filas["Presion"]
)

df_resultados.at["x̅ (Media)", "Total suma"] = round(total_media, 4)

# Reemplazar los NaN por cadenas vacías
df_resultados = df_resultados.fillna("")

# Mostrar la tabla resultante
table = tabulate(df_resultados, headers='keys', tablefmt='grid', showindex=True)
print(table)
