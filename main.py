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
import statsmodels.api as sm

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
    "x3 (Presión)": df["Presión"].round(4),
    "x3²": (df["Presión"] ** 2).round(4)
})

# Agregar la fila de sumas de Σxt
df_resultados.loc["Σxt"] = {
    "y (Oxido_nitroso)": suma_columnas["Oxido_nitroso"],
    "y²": "",
    "x1 (Humedad)": suma_columnas["Humedad"],
    "x1²": "",
    "x2 (Temperatura)": suma_columnas["Temperatura"],
    "x2²": "",
    "x3 (Presión)": suma_columnas["Presión"],
    "x3²": "",
}

# Añadir el total de las sumas en "Σxt"
df_resultados.at["Σxt", "Total suma"] = round(
    suma_columnas["Oxido_nitroso"] + suma_columnas["Humedad"] +
    suma_columnas["Temperatura"] + suma_columnas["Presión"], 4
)

# Agregar la fila de Σxt²
df_resultados.loc["Σxt²"] = {
    "y (Oxido_nitroso)": "",
    "y²": suma_cuadrados["Oxido_nitroso"],
    "x1 (Humedad)": "",
    "x1²": suma_cuadrados["Humedad"],
    "x2 (Temperatura)": "",
    "x2²": suma_cuadrados["Temperatura"],
    "x3 (Presión)": "",
    "x3²": suma_cuadrados["Presión"],
}

# Añadir el total de los cuadrados en "Σxt²"
df_resultados.at["Σxt²", "Total suma"] = round(
    suma_cuadrados["Oxido_nitroso"] + suma_cuadrados["Humedad"] +
    suma_cuadrados["Temperatura"] + suma_cuadrados["Presión"], 4
)

# Calcular el cuadrado de las sumas totales
suma_totales_cuadrado = {
    "y (Oxido_nitroso)": suma_columnas["Oxido_nitroso"] ** 2,
    "x1 (Humedad)": suma_columnas["Humedad"] ** 2,
    "x2 (Temperatura)": suma_columnas["Temperatura"] ** 2,
    "x3 (Presión)": suma_columnas["Presión"] ** 2
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
    "x3 (Presión)": round(suma_totales_cuadrado["x3 (Presión)"], 4),
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
    "x3 (Presión)": cantidad_filas["Presión"],
    "x3²": "",
}

# Añadir el total de las 'n' en "n"
total_nt = cantidad_filas["Oxido_nitroso"] + cantidad_filas["Humedad"] + cantidad_filas["Temperatura"] + cantidad_filas["Presión"]
df_resultados.at["nt", "Total suma"] = int(total_nt)  # Convierte el total a entero


# Crear la fila '(Σxt)² / nt' dividiendo cada suma total al cuadrado entre su respectivo 'nt'
df_resultados.loc["(Σxt)² / nt"] = {
    "y (Oxido_nitroso)": round(suma_totales_cuadrado["y (Oxido_nitroso)"] / cantidad_filas["Oxido_nitroso"], 4),
    "y²": "",
    "x1 (Humedad)": round(suma_totales_cuadrado["x1 (Humedad)"] / cantidad_filas["Humedad"], 4),
    "x1²": "",
    "x2 (Temperatura)": round(suma_totales_cuadrado["x2 (Temperatura)"] / cantidad_filas["Temperatura"], 4),
    "x2²": "",
    "x3 (Presión)": round(suma_totales_cuadrado["x3 (Presión)"] / cantidad_filas["Presión"], 4),
    "x3²": "",
}

# Calcular el total correcto de (Σxt)² / nt
total_xt2_n = (
    suma_totales_cuadrado["y (Oxido_nitroso)"] / cantidad_filas["Oxido_nitroso"] +
    suma_totales_cuadrado["x1 (Humedad)"] / cantidad_filas["Humedad"] +
    suma_totales_cuadrado["x2 (Temperatura)"] / cantidad_filas["Temperatura"] +
    suma_totales_cuadrado["x3 (Presión)"] / cantidad_filas["Presión"]
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
    "x3 (Presión)": round(suma_columnas["Presión"] / cantidad_filas["Presión"], 4),
    "x3²": "",
}

# Calcular el total de la media sumando los valores de cada columna
total_media = (
    suma_columnas["Oxido_nitroso"] / cantidad_filas["Oxido_nitroso"] +
    suma_columnas["Humedad"] / cantidad_filas["Humedad"] +
    suma_columnas["Temperatura"] / cantidad_filas["Temperatura"] +
    suma_columnas["Presión"] / cantidad_filas["Presión"]
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

# Parámetros de la distribución F
df1 = gl_tratamiento  # Grados de libertad del tratamiento
df2 = gl_error        # Grados de libertad del error
x = np.linspace(0, Ftab + 3, 1000)  # Rango de valores F
y = stats.f.pdf(x, df1, df2)  # Distribución F

# Crear el gráfico
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Distribución F", color="blue")

# Área de aceptación (F <= Ftab)
x_accept = np.linspace(0, Ftab, 500)
y_accept = stats.f.pdf(x_accept, df1, df2)
plt.fill_between(x_accept, y_accept, color="lightblue", alpha=0.6, label="Región de Aceptación")

# Área de rechazo (F > Ftab)
x_reject = np.linspace(Ftab, Ftab + 3, 500)
y_reject = stats.f.pdf(x_reject, df1, df2)
plt.fill_between(x_reject, y_reject, color="red", alpha=0.6, label="Región de Rechazo")

# Línea vertical en Ftab
plt.axvline(Ftab, color="black", linestyle="dashed", label=f"F tabular = {round(Ftab, 4)}")

# Línea vertical en F observado
plt.axvline(f, color="green", linestyle="dashed", label=f"F observado = {round(f, 4)}")


plt.xlabel("Valor F")
plt.ylabel("Densidad de Probabilidad")
plt.title("Distribución F de Fisher con Regiones de Aceptación y Rechazo")
plt.legend()
plt.grid()

# Mostrar el gráfico
plt.show()

########################################################################################################################

# Calcular las medias de cada grupo
x_oxido_nitroso = df_resultados.at["x̅ (Media)", "y (Oxido_nitroso)"]
x_humedad = df_resultados.at["x̅ (Media)", "x1 (Humedad)"]
x_temperatura = df_resultados.at["x̅ (Media)", "x2 (Temperatura)"]
x_Presión = df_resultados.at["x̅ (Media)", "x3 (Presión)"]

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
    "Presión": x_Presión
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


def generar_tabla_correlacion(df, var_x, var_y):
    df_resultado = pd.DataFrame({
        f"x1 ({var_x})": df[var_x].round(4),
        f"y1 ({var_y})": df[var_y].round(4),
        f"x1²": (df[var_x] ** 2).round(4),
        f"y1²": (df[var_y] ** 2).round(4),
        f"x1.y1": (df[var_x] * df[var_y]).round(4)
    })
    
    suma_columnas = {
        f"x1 ({var_x})": df[var_x].sum().round(4),
        f"y1 ({var_y})": df[var_y].sum().round(4),
        f"x1²": (df[var_x] ** 2).sum().round(4),
        f"y1²": (df[var_y] ** 2).sum().round(4),
        f"x1.y1": (df[var_x] * df[var_y]).sum().round(4)
    }
    
    df_resultado.loc["Σ"] = suma_columnas
    return df_resultado

# Crear una lista de pares independientes
pares_independientes = []

# Analizar la tabla de comparación de medias
for g1, g2, meandiff, hsd_val, independencia in tabla:
    if independencia == "Independiente":
        pares_independientes.append((g1, g2))

# Función para generar las tablas de correlación automáticamente
def generar_tablas_de_correlacion(df, pares):
    for g1, g2 in pares:
        print(f"Generando tabla de correlación para: {g1} vs {g2}")
        tabla_correlacion = generar_tabla_correlacion(df, g1, g2)
        print(tabulate(tabla_correlacion, headers="keys", tablefmt="grid"))
        print("\n")
        
        # Calcular la correlación para los pares
        correlacion = df[g1].corr(df[g2])
        print(f"Correlación ({g1} vs {g2}): {round(correlacion, 4)}\n")
        
        # Realizar la regresión lineal para los pares
        slope, intercept, _, _, _ = linregress(df[g1], df[g2])
        print(f"Ecuación de la recta para {g1} vs {g2}: y = {round(slope, 4)} * x + {round(intercept, 4)}\n")
        
        
        # Graficar la dispersión y la recta de regresión
        plt.figure(figsize=(8, 6))
        plt.scatter(df[g1], df[g2], color='blue', label='Datos', alpha=0.6)  # Puntos de dispersión
        plt.plot(df[g1], slope * df[g1] + intercept, color='red', label=f'Recta de regresión: y = {round(slope, 4)} * x + {round(intercept, 4)}')  # Recta de regresión
        plt.xlabel(f'{g1}')
        plt.ylabel(f'{g2}')
        plt.title(f'Dispersión y Recta de Regresión ({g1} vs {g2})')
        plt.legend()
        plt.grid(True)
        plt.show()

# Llamar a la función para generar las tablas de correlación y regresión para los pares independientes
generar_tablas_de_correlacion(df, pares_independientes)

######################################################################################################################################################

y = df["Oxido_nitroso"].values
x1 = df["Humedad"].values
x2 = df["Temperatura"].values
x3 = df["Presión"].values

# Tabla de Regresión Múltiple
dfmultiple = pd.DataFrame({
    "Óxido Nitroso (y)": y,
    "Humedad (x1)": x1,
    "Temperatura (x2)": x2,
    "Presión (x3)": x3,
    "y^2": np.square(y),
    "x1^2": np.square(x1),
    "x2^2": np.square(x2),
    "x3^2": np.square(x3),
    "y*x1": np.multiply(y, x1),
    "y*x2": np.multiply(y, x2),
    "y*x3": np.multiply(y, x3),
    "x1*x2": np.multiply(x1, x2),
    "x2*x3": np.multiply(x2, x3),
    "x1*x3": np.multiply(x1, x3)
})

sumatorias = dfmultiple.sum()
dfmultiple.loc["-------------"] = ["-" * 10] * dfmultiple.shape[1]
dfmultiple.loc["Σ"] = sumatorias

# Mostrar el DataFrame con las sumatorias
print("\nTabla de Contingencia con Datos Calculados:")
print(dfmultiple)

# Mostrar los resultados de las sumatorias
print("\n****Resultados de sumatorias en caso de que la tabla se resuma****")
print(f"Σyt: {round(sumatorias['Óxido Nitroso (y)'], 4)}")
print(f"Σx1t (Humedad): {round(sumatorias['Humedad (x1)'], 4)}")
print(f"Σx2t (Temperatura): {round(sumatorias['Temperatura (x2)'], 4)}")
print(f"Σx3t (Presión): {round(sumatorias['Presión (x3)'], 4)}")
print(f"Σy^2: {round(sumatorias['y^2'], 4)}")
print(f"Σx1^2: {round(sumatorias['x1^2'], 4)}")
print(f"Σx2^2: {round(sumatorias['x2^2'], 4)}")
print(f"Σx3^2: {round(sumatorias['x3^2'], 4)}")
print(f"Σy*x1: {round(sumatorias['y*x1'], 4)}")
print(f"Σy*x2: {round(sumatorias['y*x2'], 4)}")
print(f"Σy*x3: {round(sumatorias['y*x3'], 4)}")
print(f"Σx1*x2: {round(sumatorias['x1*x2'], 4)}")
print(f"Σx2*x3: {round(sumatorias['x2*x3'], 4)}")
print(f"Σx1*x3: {round(sumatorias['x1*x3'], 4)}")
print("\n")

import numpy as np
from tabulate import tabulate

# Función para resolver el sistema de ecuaciones usando el método de Gauss-Jordan
def gauss_jordan(A, B):
    AB = np.hstack([A, B.reshape(-1, 1)])  # Matriz ampliada [A|B]
    n = len(B)
    
    for i in range(n):
        # Hacer el pivote 1
        AB[i] = AB[i] / AB[i, i]
        
        for j in range(n):
            if i != j:
                AB[j] = AB[j] - AB[j, i] * AB[i]
    
    return AB  # Retornar la matriz ampliada resuelta

# Función para calcular la regresión
def calcular_regresion(dfmultiple):
    try:
        # Acceder a la fila de sumatorias correctamente
        sumatorias = dfmultiple.loc["Σ"]  
        n = len(y)  # Número de observaciones
        
        # Construir la matriz A y el vector B
        A = np.array([
            [n, sumatorias["Temperatura (x2)"], sumatorias["Presión (x3)"]],
            [sumatorias["Temperatura (x2)"], sumatorias["x2^2"], sumatorias["x2*x3"]],
            [sumatorias["Presión (x3)"], sumatorias["x2*x3"], sumatorias["x3^2"]]
        ])
        
        # Verificar si la matriz A es invertible
        det_A = np.linalg.det(A)
        if np.isclose(det_A, 0):
            raise ValueError("La matriz A no es invertible (det(A) = 0). El sistema no tiene solución única.")
        
        # Vector B
        B = np.array([
            sumatorias["Humedad (x1)"],
            sumatorias["x1*x2"],
            sumatorias["x1*x3"]
        ])
                                     
        
        # Mostrar la matriz ampliada [A|B]
        print("Matriz ampliada [A|B]:")
        print(tabulate(np.hstack([A, B.reshape(-1, 1)]), headers=["B0", "B1", "B2", "B"], tablefmt='grid', floatfmt='.4f'))
        
        
        # Resolver el sistema usando Gauss-Jordan
        matriz_resuelta = gauss_jordan(A, B)
        
        # Mostrar la matriz resultante
        print("\nMatriz resultante [A|B]:")
        print(tabulate(matriz_resuelta, headers=["B0", "B1", "B2", "B"], tablefmt='grid', floatfmt='.4f'))
        
        # Extraer los resultados
        resultados = matriz_resuelta[:, -1]
        
        # Mostrar resultados
        print("\nResultados:")
        print(f"B0 = {resultados[0]:.4f}")
        print(f"B1 = {resultados[1]:.4f}")
        print(f"B2 = {resultados[2]:.4f}")
        
        return resultados
    except Exception as e:
        print(f"Error al calcular la regresión: {e}")
        return None

# Calcular los coeficientes de regresión
coeficientes = calcular_regresion(dfmultiple)