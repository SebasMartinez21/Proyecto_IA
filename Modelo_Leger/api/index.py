from flask import Flask, request, render_template
import pickle 
import pandas as pd
import os

app = Flask(__name__, template_folder="../templates", static_folder="../static")
modelo = pickle.load(open(os.path.join(os.path.dirname(__file__), "../modelo/modelo0.pkl"), "rb"))

columnas = pickle.load(open(os.path.join(os.path.dirname(__file__), "../modelo/columnas0.pkl"), "rb"))

@app.route("/", methods=["GET"])
def formulario():
    return render_template("formulario.html")

@app.route("/predecir", methods=["POST"])
def predecir():
    datos = { 
        "Sexo": request.form["Sexo"],
        "IMC_categoria": request.form["IMC_categoria"],
        "Salto_Largo": request.form["Salto_Largo"],
        "30_m": request.form["30_m"],
        "IMC": float(request.form["IMC"]),
        "IA": float(request.form["IA"]),
        "PR": float(request.form["PR"])
    }

    df = pd.DataFrame([datos])
    df_encoded = pd.get_dummies(df)
    for col in columnas:
        if col not in df_encoded:
            df_encoded[col] = 0
    prediccion = modelo.predict(df_encoded[columnas])[0]
    return render_template("resultado.html", prediccion=round(prediccion, 2))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
