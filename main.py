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
from scipy.stats import linregress

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
print("\n")
########################################################################################################################

# Nivel de significancia
nivel_significancia = 0.95
alfa = 1 - nivel_significancia
print(f"α (alfa): {round(alfa, 4)}\n")

# Número de tratamientos (columnas)
t = len(df.columns)

# Grados de libertad del tratamiento
gl_tratamiento = t - 1
print(f"gl(tratamiento): {gl_tratamiento}\n")

# Grados de libertad del error
Σnt_4col = df_resultados.at["nt", "Total suma"]
gl_error = Σnt_4col - t  
print(f"gl(error): {round(gl_error, 4)}\n")

# Factor de corrección (C)
Σxt_4col = df_resultados.at["Σxt", "Total suma"]
c = (Σxt_4col ** 2) / Σnt_4col 
print(f"Factor de corrección (C): {round(c, 4)}\n")

# Suma Total de Cuadrados (SCT)
Σxt2_4col = df_resultados.at["Σxt²", "Total suma"]
sct = Σxt2_4col - c 
print(f"Suma Total de Cuadrados (SCT): {round(sct, 4)}\n")

# Suma Cuadrada de Tratamiento (SCTR)
ΣxtcuaN_4col = df_resultados.at["(Σxt)² / nt", "Total suma"]
sctr = ΣxtcuaN_4col - c
print(f"Suma Cuadrada de Tratamiento (SCTR): {round(sctr, 4)}\n")

# Suma Cuadrada de Error (SCE)
sce = sct - sctr
print(f"Suma Cuadrada de Error (SCE): {round(sce, 4)}\n")

# Calcular n - 1 (grados de libertad totales)
nt_1 = Σnt_4col - 1
print(f"n - 1: {round(nt_1, 4)}\n")

# Media Cuadrada de Tratamiento (MCTR) y Media Cuadrada de Error (MCE)
mctr = sctr / gl_tratamiento
mce = sce / gl_error
print(f"Media Cuadrada de Tratamiento (MCTR): {round(mctr, 4)}")
print(f"Media Cuadrada de Error (MCE): {round(mce, 4)}\n")

# Razón de Variación (Fisher)
f = mctr / mce
print(f"F (Razón de Variación de Fisher): {round(f, 4)}\n")

# Razón de Variación (Fisher)
f_rv = f

# Crear el DataFrame de la fuente de variación
fuente_variacion = pd.DataFrame({
    "Fuentes de variación": [
        "│ (Tratamiento) │", 
        "│ (Error)       │", 
        "│ (Total)       │"
    ],
    "SC": [
        f"│ ({round(sctr, 4)}) │", 
        f"│ ({round(sce, 4)}) │", 
        f"│ ({round(sct, 4)}) │"
    ],
    "gl": [
        f"│ ({round(gl_tratamiento, 4)}) │", 
        f"│ ({round(gl_error, 4)}) │", 
        f"│ ({round(nt_1, 4)}) │"
    ],
    "MC": [
        f"│ ({round(mctr, 4)}) │", 
        f"│ ({round(mce, 4)}) │", 
        "│ (********) │"
    ],
    "F (RV)": [
        f"│ ({round(f_rv, 4)}) │", 
        "│ (********) │", 
        "│ (********) │"
    ]
})

# Imprimo la tabla con tabulate
print("\nFuente de Variación:")
print(tabulate(fuente_variacion, headers="keys", tablefmt="grid"))

# Cálculo de F tabular
Ftab = stats.f.ppf(1 - alfa, gl_tratamiento, gl_error)
print(f"\nF tabular: {round(Ftab, 4)}\n")

# Comparación y decisión
decision = "Rechazar H₀ (Existe diferencia significativa)" if f > Ftab else "Aceptar H₀ (No hay diferencia significativa)"
print(f"Decisión: {decision}")

########################################################################################################################

# Calcular las medias de cada grupo
x_oxido_nitroso = df_resultados.at["x̅ (Media)", "y (Oxido_nitroso)"]
x_humedad = df_resultados.at["x̅ (Media)", "x1 (Humedad)"]
x_temperatura = df_resultados.at["x̅ (Media)", "x2 (Temperatura)"]
x_presion = df_resultados.at["x̅ (Media)", "x3 (Presion)"]

num_grupos = t  # Número de grupos (columnas)

# Valor crítico q para HSD
q = studentized_range.ppf(1 - alfa, num_grupos, gl_error)

# Calculo DHS (Diferencia Honestamente Significativa)
nt_oxido_nitroso = cantidad_filas["Oxido_nitroso"]
hsd = q * np.sqrt(mce / nt_oxido_nitroso)  

print()
print(f"Valor crítico q: {round(q, 4)}")
print(f"Diferencia Honestamente Significativa (HSD): {round(hsd, 4)}")

# Medias de cada grupo
medias = {
    "Óxido Nitroso": x_oxido_nitroso,
    "Humedad": x_humedad,
    "Temperatura": x_temperatura,
    "Presión": x_presion
}

# Lista de pares para comparar
pares = [
    ("Óxido Nitroso", "Humedad"),
    ("Óxido Nitroso", "Temperatura"),
    ("Óxido Nitroso", "Presión"),
    ("Humedad", "Temperatura"),
    ("Humedad", "Presión"),
    ("Temperatura", "Presión")
]

# Crear la tabla de resultados
tabla = []
for g1, g2 in pares:
    meandiff = medias[g1] - medias[g2]
    independencia = "Independiente" if meandiff > hsd or meandiff > -hsd else "Dependiente"
    tabla.append([g1, g2, f"{meandiff:.4f}", f"{hsd:.4f}", independencia])

# Imprimir la tabla con tabulate
print("\nComparación de Medias - Prueba de Tukey\n")
headers = ["Grupo 1", "Grupo 2", "Diferencia", "DHS", "Independencia"]
print(tabulate(tabla, headers=headers, tablefmt="grid"))
print("\n")
######################################################################################################################

"""tengo que hacer la tabla x1 y1 x elevado al 2 y elevado al cuadrado y x.y esa tabla debo hacerla 
por cada par que sea independiente es decir debo hacer dos tablas con la misma estructura de las tablas que hemos trabajado esto
para hallar la correlacion y la recta lineal de ajuste y luego hacer los calulos de correlacion y regresion  """  
    
    # Funcion para crear tablas de x1, y1, x^2, y^2, x.y
def generar_tabla_correlacion(df, var_x, var_y):
    
    df_resultado = pd.DataFrame({
        f"x1 ({var_x})": df[var_x].round(4),
        f"y1 ({var_y})": df[var_y].round(4),
        f"x1²": (df[var_x] ** 2).round(4),
        f"y1²": (df[var_y] ** 2).round(4),
        f"x1.y1": (df[var_x] * df[var_y]).round(4)
    })
    
    # Calculamos la suma de x1, y1, x1², y1², x1.y1
    suma_columnas = {
        f"x1 ({var_x})": df[var_x].sum().round(4),
        f"y1 ({var_y})": df[var_y].sum().round(4),
        f"x1²": (df[var_x] ** 2).sum().round(4),
        f"y1²": (df[var_y] ** 2).sum().round(4),
        f"x1.y1": (df[var_x] * df[var_y]).sum().round(4)
    }

    # Añadimos las sumas al final de la tabla
    df_resultado.loc["Σ"] = suma_columnas

    return df_resultado

# Genero las tablas para los pares independientes
tabla_humedad_presion = generar_tabla_correlacion(df, "Humedad", "Presion")
tabla_temperatura_presion = generar_tabla_correlacion(df, "Temperatura", "Presion")

# Imprimo las tablas generadas
print("Tabla Humedad vs Presión:")
print(tabulate(tabla_humedad_presion, headers="keys", tablefmt="grid"))
print("\n")

print("Tabla Temperatura vs Presión:")
print(tabulate(tabla_temperatura_presion, headers="keys", tablefmt="grid"))
print("\n")

# Calculo la correlacion para ambos pares
correlacion_humedad_presion = df["Humedad"].corr(df["Presion"])
correlacion_temperatura_presion = df["Temperatura"].corr(df["Presion"])


print(f"Correlación (Humedad vs Presión): {round(correlacion_humedad_presion, 4)}")
print(f"Correlación (Temperatura vs Presión): {round(correlacion_temperatura_presion, 4)}")
print("\n")

# Realizo la regresion lineal para ambos pares
# Humedad vs Presion
slope_hum, intercept_hum, _, _, _ = linregress(df["Humedad"], df["Presion"])
# Temperatura vs Presion
slope_temp, intercept_temp, _, _, _ = linregress(df["Temperatura"], df["Presion"])

# Muestror la ecuacion de la recta de ajuste
print(f"Ecuación de la recta para Humedad vs Presión: y = {round(slope_hum, 4)} * x + {round(intercept_hum, 4)}")
print(f"Ecuación de la recta para Temperatura vs Presión: y = {round(slope_temp, 4)} * x + {round(intercept_temp, 4)}")


# Función para graficar la dispersión y la recta de regresión
def graficar_regresion(df, var_x, var_y, slope, intercept, title):
    plt.figure(figsize=(8, 6))
    
    # Graficar puntos de dispersión
    plt.scatter(df[var_x], df[var_y], color='blue', label='Datos', alpha=0.7)

    # Graficar la recta de regresión
    plt.plot(df[var_x], slope * df[var_x] + intercept, color='red', label=f'Recta de ajuste: y = {round(slope, 4)} * x + {round(intercept, 4)}')
    
    # Títulos y etiquetas
    plt.title(title)
    plt.xlabel(var_x)
    plt.ylabel(var_y)
    
    # Mostrar leyenda
    plt.legend()

    # Mostrar el gráfico
    plt.grid(True)
    plt.show()

# Graficar para Humedad vs Presión
graficar_regresion(df, "Humedad", "Presion", slope_hum, intercept_hum, "Humedad vs Presión")

# Graficar para Temperatura vs Presión
graficar_regresion(df, "Temperatura", "Presion", slope_temp, intercept_temp, "Temperatura vs Presión")
