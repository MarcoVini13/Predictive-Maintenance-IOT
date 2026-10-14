import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train():
    # 1. Cargar datos
    df = pd.read_csv('data/sensor_data.csv')
    
    X = df[['temperature', 'vibration', 'pressure', 'operating_hours']]
    y = df['failure']
    
    # 2. Dividir entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluar
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"🎯 Precisión del modelo: {acc * 100:.2f}%")
    print("\nReporte de Clasificación:\n", classification_report(y_test, predictions))
    
    # 5. Guardar el modelo
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/maintenance_model.pkl')
    print("✅ Modelo guardado en 'models/maintenance_model.pkl'")

if __name__ == "__main__":
    train()
