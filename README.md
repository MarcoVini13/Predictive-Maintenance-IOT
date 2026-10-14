# ⚙️ Industrial Predictive Maintenance Dashboard (IoT + Machine Learning)

An end-to-end Machine Learning solution for predictive maintenance and real-time failure analysis in industrial machinery. Built with Python, Scikit-Learn, Streamlit, and Plotly.

---

## 📌 Project Overview
Unplanned downtime in manufacturing environments leads to significant operational losses. This project simulates real-time industrial telemetry (temperature, vibration levels, pressure, and operational hours) and uses a **Random Forest Classifier** to predict machine failure risk before critical breakdowns occur.

Key Features:
* **Interactive Live Simulation:** Sliders to emulate real-time sensor readings and evaluate instant failure probabilities.
* **Predictive ML Engine:** Trained model predicting failure risks based on physical degradation patterns.
* **Historical Telemetry Visualization:** Interactive scatter plots showing correlation between sensor parameters and machine status.

---

## 🛠️ Tech Stack
* **Language:** Python 3.13+
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn, Joblib
* **Data Visualization:** Plotly Express
* **Web Interface:** Streamlit

---

## 📂 Project Structure
```text
predictive-maintenance-iot/
├── data/
│   └── sensor_data.csv          # Generated industrial sensor dataset
├── models/
│   └── maintenance_model.pkl    # Trained Random Forest Model
├── src/
│   ├── generate_data.py         # Telemetry simulation script
│   └── train_model.py           # Machine learning training pipeline
├── app.py                       # Streamlit web dashboard
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 🚀 How to Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/MarcoVini13/predictive-maintenance-iot.git
cd predictive-maintenance-iot
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Generate synthetic telemetry data:**
```bash
python src/generate_data.py
```

4. **Train the ML model:**
```bash
python src/train_model.py
```

5. **Launch the interactive dashboard:**
```bash
python -m streamlit run app.py
```

---

👤 **Author:** Marco Vinicio Almendarez Briones
