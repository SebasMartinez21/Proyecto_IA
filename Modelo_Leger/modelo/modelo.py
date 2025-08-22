import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
'''
Modelo de predicción de rendimiento en Léger (Regresión)
Variable Target: 
- Léger
Variables necesarias (Features): 
- Salto largo
- 30 m lanzados (s)
- IMC
- PR (Potencia Relativa)
- IA (Indice de Agilidad)
'''

# 💎 Lectura del excel (.xlsx)
df = pd.read_excel(r"C:\Users\equipo\Desktop\Bootcamp_IA\HojadeVida&ProyectoIA\ProyectoIA\Analisis_exploratorio\data.xlsx") 

# 💎 Imprimir las columnas 
#print(df.columns)

# 💎 Imprimir la suma de los nulos por columna 
#print("\n Nulos por columnas de los datos: \n", df.isnull().sum())

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
1. ° PR (Potencia relativa) -> (Salto Largo/Peso)
2. ° IA (Indice de Agilidad) -> Altura/(10x5 (s) X Peso (kg))
'''

df["PR"] = df["Salto Largo"]/df["Peso (kg)"]

df["IA"] = df["Talla (cm)"]/(df["10 x 5 (s)"]*df["Peso (kg)"])

# 💎 Filtrar las columnas necesarias
columnas_necesarias = ["Leger", "Salto Largo", "Velocidad Maxima", "PR", "IA"]

df = df[columnas_necesarias]

print("\n Nulos por columnas de los datos: \n", df.isnull().sum())
print("\n DataFrame resultante: \n", df)

# 💎 Separar variables
x = df.drop(columns = "Leger")
y = df["Leger"]

# 💎 Codificar variables categoricas
x_encoded = pd.get_dummies(x)

# 💎 Dividir datos 
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=42)

# 💎 Entrenar modelo 
modelo = LinearRegression()
modelo.fit(x_train, y_train)

# 💎 Guardar modelo 
with open('modelo.pkl', 'wb') as f:
    pickle.dump(modelo, f)
with open('columnas.pkl', 'wb') as f:
    pickle.dump(x_encoded.columns.tolist(), f)
