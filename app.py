from flask import Flask, render_template, request, send_file
import numpy as np
import pickle
import plotly.graph_objs as go
import plotly.io as pio
from reportlab.pdfgen import canvas
import os

app = Flask(__name__)

model = pickle.load(open("hybrid_model.pkl", "rb"))
scaler = pickle.load(open("scaling.pkl", "rb"))

FEATURE_IMG = "static/feature_importance.png"
CM_IMG = "static/confusion_matrix.png"
ROC_IMG = "static/roc_curve.png"
CR_IMG = "static/classification_report.png"

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
    arr = np.array(vals).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prob = float(model.predict_proba(arr_scaled)[0][1])
    risk_text = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    labels = [
        "Age", "Anaemia", "CPK", "Diabetes", "Ejection Fraction", "High BP",
        "Platelets", "Creatinine", "Sodium", "Sex", "Smoking", "Follow-up Time"
    ]
    fig = go.Figure([go.Bar(
        x=labels,
        y=arr_scaled[0],
        marker_color="#4a6cf7"
    )])
    fig.update_layout(
        title="Patient Feature Contribution",
        xaxis_title="Features",
        yaxis_title="Scaled Values",
        template="plotly_white",
        height=460
    )
    graph_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": False}
    )
    return render_template(
        "result.html",
        probability=round(prob, 3),
        risk=risk_text,
        graph=graph_html,
        feature_img=FEATURE_IMG,
        cm_img=CM_IMG,
        roc_img=ROC_IMG,
        cr_img=CR_IMG
    )

if __name__ == "__main__":
    app.run(debug=True)
