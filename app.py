from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objs as go
import plotly.io as pio
import os
from datetime import datetime

app = Flask(__name__)

model = pickle.load(open("hybrid_model.pkl", "rb"))
scaler = pickle.load(open("scaling.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    vals = [
        float(request.form["age"]),
        float(request.form["anaemia"]),
        float(request.form["creatinine_phosphokinase"]),
        float(request.form["diabetes"]),
        float(request.form["ejection_fraction"]),
        float(request.form["high_blood_pressure"]),
        float(request.form["platelets"]),
        float(request.form["serum_creatinine"]),
        float(request.form["serum_sodium"]),
        float(request.form["sex"]),
        float(request.form["smoking"]),
        float(request.form["time"])
    ]

    arr = scaler.transform(np.array(vals).reshape(1, -1))
    prob = float(model.predict_proba(arr)[0][1])
    risk = "HIGH RISK" if prob >= 0.5 else "LOW RISK"

    labels = [
        "Age","Anaemia","CPK","Diabetes","Ejection Fraction",
        "High BP","Platelets","Creatinine","Sodium","Sex","Smoking","Time"
    ]

    fig = go.Figure([go.Bar(x=labels, y=arr[0], marker_color="#1e88e5")])
    fig.update_layout(template="plotly_white", height=400)

    graph = pio.to_html(fig, full_html=False)

    return render_template(
        "result.html",
        risk=risk,
        probability=round(prob, 3),
        graph=graph
    )

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        return "Invalid file format"

    cols = [
        "age","anaemia","creatinine_phosphokinase","diabetes",
        "ejection_fraction","high_blood_pressure","platelets",
        "serum_creatinine","serum_sodium","sex","smoking","time"
    ]

    X = scaler.transform(df[cols])
    probs = model.predict_proba(X)[:, 1]

    df["risk_probability"] = probs
    df["risk_prediction"] = ["HIGH RISK" if p >= 0.5 else "LOW RISK" for p in probs]

    high = (df["risk_prediction"] == "HIGH RISK").sum()
    low = (df["risk_prediction"] == "LOW RISK").sum()

    filename = f"heart_failure_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.join("static", filename)
    df.to_csv(path, index=False)

    return render_template(
        "batch_result.html",
        total=len(df),
        high=high,
        low=low,
        filename=filename
    )

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join("static", filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
