import pandas as pd # Manipulación y análisis de datos en estructuras tipo DataFrame
import scipy.stats as stats # Módulo de estadísticas en SciPy, útil para pruebas estadísticas y distribuciones
import statsmodels.api as sm # Biblioteca para modelos estadísticos, incluye regresión y pruebas estadísticas
from statsmodels.formula.api import ols # Permite realizar Análisis de Varianza (ANOVA) y modelos de regresión con fórmulas
from statsmodels.stats.multicomp import pairwise_tukeyhsd # Prueba de comparaciones múltiples de Tukey para ANOVA
from tabulate import tabulate # Formatea datos en tablas bien organizadas para impresión en consola
import numpy as np  # Cálculos numéricos eficientes con arreglos y funciones matemáticas avanzadas
import matplotlib.pyplot as plt # Biblioteca de visualización para gráficos estáticos
import seaborn as sns # Biblioteca de visualización basada en Matplotlib con gráficos más estilizados
from scipy.stats import studentized_range # Distribución del rango studentizado, util en pruebas de comparaciones múltiples
from scipy.stats import linregress # Realiza regresión lineal simple y proporciona métricas asociadas
from itertools import combinations
#############################################################################################################################################

# aqui se carga el archivo CSV en el DataFrame
df = pd.read_csv('data/datos_estudio.csv')

# Inicializar diccionarios para almacenar sumas, sumas de cuadrados y conteos
suma_columnas = {}
suma_cuadrados = {}
cantidad_filas = {}

# Realizar cálculos en un solo bucle
for columna in df.columns:
    suma_columnas[columna] = round(df[columna].sum(), 4)
    suma_cuadrados[columna] = round((df[columna] ** 2).sum(), 4)
    cantidad_filas[columna] = len(df[columna])

# Creo el DataFrame con los valores originales y sus cuadrados

LISTA_COLUMNAS = df.columns.tolist()
info_columnas = {}

# Llenar el diccionario de información
for i, columna in enumerate(LISTA_COLUMNAS):
    if i == 0: 
        info_columnas[f'y ({columna})'] = df[columna].round(4)
        info_columnas['y²'] = (df[columna] ** 2).round(4)
    else: 
        info_columnas[f'x{i} ({columna})'] = df[columna].round(4)
        info_columnas[f'x{i}²'] = (df[columna] ** 2).round(4)

# Crear el DataFrame
df_resultados = pd.DataFrame(info_columnas)

# Definir las operaciones a realizar
operaciones = {
    "Σxt": lambda: {
        f'y ({LISTA_COLUMNAS[0]})': suma_columnas[LISTA_COLUMNAS[0]],
        'y²': '',
        **{f'x{i} ({columna})': suma_columnas[columna] for i, columna in enumerate(LISTA_COLUMNAS[1:], start=1)},
        **{f'x{i}²': '' for i in range(1, len(LISTA_COLUMNAS))}
    },
    "Σxt²": lambda: {
        f'y ({LISTA_COLUMNAS[0]})': '',
        'y²': suma_cuadrados[LISTA_COLUMNAS[0]],
        **{f'x{i} ({columna})': '' for i, columna in enumerate(LISTA_COLUMNAS[1:], start=1)},
        **{f'x{i}²': suma_cuadrados[columna] for i, columna in enumerate(LISTA_COLUMNAS[1:], start=1)}
    },
    "(Σxt)²": lambda: {
        f'y ({LISTA_COLUMNAS[0]})': round(suma_columnas[LISTA_COLUMNAS[0]] ** 2, 4),
        'y²': '',
        **{f'x{i} ({columna})': round(suma_columnas[columna] ** 2, 4) for i, columna in enumerate(LISTA_COLUMNAS[1:], start=1)},
        **{f'x{i}²': '' for i in range(1, len(LISTA_COLUMNAS))}
    },
    "nt": lambda: {
        f'y ({LISTA_COLUMNAS[0]})': cantidad_filas[LISTA_COLUMNAS[0]],
        'y²': '',
        **{f'x{i} ({columna})': cantidad_filas[columna] for i, columna in enumerate(LISTA_COLUMNAS[1:], start=1)},
        **{f'x{i}²': '' for i in range(1, len(LISTA_COLUMNAS))}
    },
    "(Σxt)² / nt": lambda: {
        f'y ({LISTA_COLUMNAS[0]})': round(suma_columnas[LISTA_COLUMNAS[0]] ** 2 / cantidad_filas[LISTA_COLUMNAS[0]], 4),
        'y²': '',
        **{f'x{i} ({columna})': round(suma_columnas[columna] ** 2 / cantidad_filas[columna], 4) for i, columna in enumerate(LISTA_COLUMNAS[1:], start=1)},
        **{f'x{i}²': '' for i in range(1, len(LISTA_COLUMNAS))}
    },
    "x̅ (Media)": lambda: {
        f'y ({LISTA_COLUMNAS[0]})': round(suma_columnas[LISTA_COLUMNAS[0]] / cantidad_filas[LISTA_COLUMNAS[0]], 4),
        'y²': '',
        **{f'x{i} ({columna})': round(suma_columnas[columna] / cantidad_filas[columna], 4) for i, columna in enumerate(LISTA_COLUMNAS[1:], start=1)},
        **{f'x{i}²': '' for i in range(1, len(LISTA_COLUMNAS))}
    }
}

# Agregar las filas al DataFrame
for operacion, calculo in operaciones.items():
    df_resultados.loc[operacion] = calculo()
    # Calcular y agregar el total de cada operación
    if operacion in ["Σxt", "Σxt²"]:
        df_resultados.at[operacion, "Total suma"] = round(sum(suma_columnas.values()) if operacion == "Σxt" else sum(suma_cuadrados.values()), 4)
    elif operacion == "(Σxt)²":
        df_resultados.at[operacion, "Total suma"] = round(sum(suma_columnas[columna] ** 2 for columna in LISTA_COLUMNAS), 4)
    elif operacion == "nt":
        df_resultados.at[operacion, "Total suma"] = sum(cantidad_filas.values())
    elif operacion == "(Σxt)² / nt":
        df_resultados.at[operacion, "Total suma"] = round(sum(suma_columnas[columna] ** 2 / cantidad_filas[columna] for columna in LISTA_COLUMNAS), 4)
    elif operacion == "x̅ (Media)":
        df_resultados.at[operacion, "Total suma"] = round(sum(suma_columnas[columna] / cantidad_filas[columna] for columna in LISTA_COLUMNAS), 4)

# Reemplazo todos los NaN por cadenas vacías
df_resultados = df_resultados.fillna("")

# Muestro la tabla resultante
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
        f"│ ({round(f, 4)}) │", 
        "│ (********) │", 
        "│ (********) │"
    ]
})

# Imprimo la tabla con tabulate
print("\nFuente de Variación:")
print(tabulate(fuente_variacion, headers="keys", tablefmt="grid"))

# Calculo de F tabular
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

# Etiquetas y título
plt.xlabel("Valor F")
plt.ylabel("Densidad de Probabilidad")
plt.title("Distribución F de Fisher con Regiones de Aceptación y Rechazo")
plt.legend()
plt.grid()

# Muestro el gráfico
plt.show()

########################################################################################################################

# Obtener dinámicamente las medias de cada variable
medias = {col: df_resultados.at["x̅ (Media)", f"y ({col})"] if i == 0 else df_resultados.at["x̅ (Media)", f"x{i} ({col})"]
          for i, col in enumerate(LISTA_COLUMNAS)}

# Obtener número de grupos (columnas en el DataFrame)
num_grupos = len(medias)

# Calcular el valor crítico q para la prueba de Tukey
q = studentized_range.ppf(1 - alfa, num_grupos, gl_error)

# Calcular HSD (Diferencia Honestamente Significativa)
nt_oxido_nitroso = cantidad_filas[LISTA_COLUMNAS[0]]  # Suponiendo que todas las columnas tienen el mismo n
hsd = q * np.sqrt(mce / nt_oxido_nitroso)

print(f"\nValor crítico q: {round(q, 4)}")
print(f"Diferencia Honestamente Significativa (HSD): {round(hsd, 4)}\n")

# Generar todas las combinaciones posibles de pares de variables
pares = list(combinations(medias.keys(), 2))

# Crear la tabla de comparación de medias
tabla = []
for g1, g2 in pares:
    meandiff = medias[g1] - medias[g2]
    independencia = "Independiente" if meandiff > hsd or meandiff > -hsd else "Dependiente"
    tabla.append([g1, g2, f"{meandiff:.4f}", f"{hsd:.4f}", independencia])

# Imprimir la tabla de comparación de medias usando tabulate
headers = ["Grupo 1", "Grupo 2", "Diferencia", "DHS", "Independencia"]
print("\nComparación de Medias - Prueba de Tukey\n")
print(tabulate(tabla, headers=headers, tablefmt="grid"))
print("\n")

######################################################################################################################

def generar_tabla_correlacion(df, var_x, var_y):
    # Crear un DataFrame con las columnas necesarias
    df_resultado = pd.DataFrame({
        f"x1 ({var_x})": df[var_x].round(4),
        f"y1 ({var_y})": df[var_y].round(4),
        f"x1²": (df[var_x] ** 2).round(4),
        f"y1²": (df[var_y] ** 2).round(4),
        f"x1.y1": (df[var_x] * df[var_y]).round(4)
    })
    
    # Calcular las sumas de las columnas
    suma_columnas = {
        f"x1 ({var_x})": df[var_x].sum().round(4),
        f"y1 ({var_y})": df[var_y].sum().round(4),
        f"x1²": (df[var_x] ** 2).sum().round(4),
        f"y1²": (df[var_y] ** 2).sum().round(4),
        f"x1.y1": (df[var_x] * df[var_y]).sum().round(4)
    }
    
    # Agregar la fila de sumas al DataFrame
    df_resultado.loc["Σ"] = suma_columnas
    return df_resultado

# Crear una lista de pares independientes
pares_independientes = []

# Aquí se analiza la tabla de comparación de medias
for g1, g2, meandiff, hsd_val, independencia in tabla:
    if independencia == "Independiente":
        pares_independientes.append((g1, g2))

# Función para generar las tablas de correlación
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

# Llamar a la función para generar las tablas de correlación
generar_tablas_de_correlacion(df, pares_independientes)

######################################################################################################################################################

y = df["Oxido_nitroso"].values
x1 = df["Humedad"].values
x2 = df["Temperatura"].values
x3 = df["Presión"].values

# Tabla de Regresion Multiple
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

# Calculo las sumatorias
sumatorias = dfmultiple.sum()
dfmultiple.loc["Σ"] = sumatorias

# Mostrar el DataFrame con las sumatorias
print("\nTabla de Contingencia con Datos Calculados:")
print(dfmultiple)

# Resultados de las sumatorias
print("\n**** Resultados de sumatorias ****")
sumatorias_resultados = {
    "Σyt": round(sumatorias['Óxido Nitroso (y)'], 4),
    "Σx1t (Humedad)": round(sumatorias['Humedad (x1)'], 4),
    "Σx2t (Temperatura)": round(sumatorias['Temperatura (x2)'], 4),
    "Σx3t (Presión)": round(sumatorias['Presión (x3)'], 4),
    "Σy^2": round(sumatorias['y^2'], 4),
    "Σx1^2": round(sumatorias['x1^2'], 4),
    "Σx2^2": round(sumatorias['x2^2'], 4),
    "Σx3^2": round(sumatorias['x3^2'], 4),
    "Σy*x1": round(sumatorias['y*x1'], 4),
    "Σy*x2": round(sumatorias['y*x2'], 4),
    "Σy*x3": round(sumatorias['y*x3'], 4),
    "Σx1*x2": round(sumatorias['x1*x2'], 4),
    "Σx2*x3": round(sumatorias['x2*x3'], 4),
    "Σx1*x3": round(sumatorias['x1*x3'], 4)
}

# Preparar datos para tabular
tabla = [[key, value] for key, value in sumatorias_resultados.items()]

# Mostrar resultados de sumatorias con tabulado
print(tabulate(tabla, headers=["Descripción", "Valor"], tablefmt="grid"))
print("\n")
#################################################################################################################3

# Funcion para resolver el sistema de ecuaciones usando el metodo de Gauss-Jordan
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

# Funcion para calcular la regresion
def calcular_regresion(dfmultiple):
    try:
        # Acceder a la fila de sumatorias
        sumatorias = dfmultiple.loc["Σ"]  
        n = len(y)  # Numero de observaciones
        
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
        
        # Muestro la matriz resultante
        print("\nMatriz resultante [A|B]:")
        print(tabulate(matriz_resuelta, headers=["B0", "B1", "B2", "B"], tablefmt='grid', floatfmt='.4f'))
        
        # Extraer los resultados
        resultados = matriz_resuelta[:, -1]
        
        # Muestro los resultados
        print("\nResultados:")
        print(f"B0 = {resultados[0]:.4f}")
        print(f"B1 = {resultados[1]:.4f}")
        print(f"B2 = {resultados[2]:.4f}")
        
         # Construcción de la ecuación de regresión
        columnas_independientes = ["Humedad (x1)", "Temperatura (x2)", "Presión (x3)"]
        ecuacion = "y = " + " + ".join([f"{resultados[i]:.4f}*{columnas_independientes[i]}" for i in range(len(resultados))])
        print("\nRecta de regresión múltiple:")
        print(ecuacion)

        return resultados
    
        return resultados
    except Exception as e:
        print(f"Error al calcular la regresión: {e}")
        return None

# Calculo los coeficientes de regresion
coeficientes = calcular_regresion(dfmultiple)
print("\n")