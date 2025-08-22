import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
'''
Modelo de predicción de rendimiento en Léger (Regresión)
Variable Target: 
- Léger
Variables necesarias (Features): 
- Velocidad maxima
- VR (Velocidad Relativa)
- IVT (Indice de Velocidad-Talla)
'''

# 💎 Lectura del excel (.xlsx)
df = pd.read_excel(r"C:\Users\equipo\Desktop\Bootcamp_IA\HojadeVida&ProyectoIA\ProyectoIA\Analisis_exploratorio\data.xlsx") 

# 💎 Rellenar los valores de Leger con la media.
if "Leger" in df:
    df["Leger"] = df["Leger"].fillna(df["Leger"].mean())

# 💎 No hay columnas con exceso de nulos que puedan ser borradas...

# 💎 Cambiar el nombre de la columna Velocidad Máxima (Course Navette - km h^-1) y Edad (años cumplidos)
#     por Velocida Maxima y Edad respectivamente.
df = df.rename(columns={"Velocidad Máxima (Course Navette - km h^-1)" : "Velocidad Maxima"})
df = df.rename(columns={"Edad (años cumplidos)": "Edad"})

# 💎 Rellenar los valores de Velocidad Máxima con la media.
if "Velocidad Maxima" in df:
    df["Velocidad Maxima"] = df["Velocidad Maxima"].fillna(df["Velocidad Maxima"].mean())

# Imprimir la suma de los nulos por columna.
#print("\n Nulos por columnas de los datos: \n", df.isnull().sum())

'''
-------------------- Agregar columnas de: --------------------
1. ° IMC (Indice de Masa Corporal) -> Peso/((Talla/100)^2)
2. ° VR (Velocidad Relativa) -> Velocidad Maxima/Peso
3. ° IVT (Indice de Velocidad-Talla) -> Velocidad Maxima/Talla
4. ° Clasificación de IMC
df["IMC"] = df["Peso (kg)"]/((df["Talla (cm)"]/100)**2)

df["VR"] = df["Velocidad Maxima"]/df["Peso (kg)"]

df["IVT"] = df["Velocidad Maxima"]/df["Talla (cm)"]
'''

'''
-------------------- Agregar columnas de: --------------------
1. ° IMC (Indice de Masa Corporal) -> Peso/((Talla/100)^2)
2. ° PR (Potencia Relativa) -> Velocidad Maxima/Peso
3. ° IA (Indice de Agilidad) -> Velocidad Maxima/Talla
4. ° Clasificación de IMC
'''

df["IMC"] = df["Peso (kg)"]/((df["Talla (cm)"]/100)**2)

df["PR"] = df["Salto Largo"]/df["Peso (kg)"]

df["IA"] = df["Talla (cm)"]/(df["10 x 5 (s)"]*df["Peso (kg)"])

# 💎 Clasificación de IMC (Nueva columna)
df["IMC_categoria"] = pd.cut(
    df["IMC"],
    bins = [0, 18.5, 24.9, 29.9, np.inf],
    labels = ["Bajo peso", "Peso normal", "Sobrepeso", "Obesidad"]
)

# 💎 Filtrar las columnas necesarias
columnas_necesarias = ["Sexo", "Leger", "Salto Largo", "30 m lanzados (s)", "IMC", "IA", "PR", "IMC_categoria"]

df = df[columnas_necesarias]

print("\n Nulos por columnas de los datos: \n", df.isnull().sum())
print("\n DataFrame resultante: \n", df)

# 💎 Separar variables
x = df.drop(columns = "Leger")
y = df["Leger"]

# 💎 Codificar variables categoricas
x_encoded = pd.get_dummies(x)
print(x_encoded)
print("\n", x_encoded.columns)

# 💎 Dividir datos 
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=42)

# 💎 Entrenar modelo 
modelo = LinearRegression()
modelo.fit(x_train, y_train)

# 💎 Guardar modelo 
with open('modelo0.pkl', 'wb') as f:
    pickle.dump(modelo, f)
with open('columnas0.pkl', 'wb') as f:
    pickle.dump(x_encoded.columns.tolist(), f)
