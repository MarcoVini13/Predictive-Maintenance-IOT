import pandas as pd
import numpy as np
import os

def generate_sensor_data(samples=1200):
    np.random.seed(42)
    
    # Simulación de telemetría de sensores
    temperature = np.random.normal(loc=70, scale=12, size=samples)  # °C
    vibration = np.random.normal(loc=3.5, scale=1.2, size=samples)   # mm/s
    pressure = np.random.normal(loc=95, scale=10, size=samples)     # PSI
    operating_hours = np.random.uniform(low=100, high=8000, size=samples)
    
    # Lógica física para simular una falla (Fail = 1, Normal = 0)
    # Falla si la temperatura es alta + alta vibración o exceso de horas
    failure_score = (
        (temperature > 88) * 0.4 +
        (vibration > 5.2) * 0.35 +
        (pressure < 75) * 0.15 +
        (operating_hours > 6500) * 0.2
    )
    
    # Convertir score en clasificación binaria (0 o 1)
    failure = (failure_score + np.random.normal(0, 0.1, samples) > 0.45).astype(int)
    
    df = pd.DataFrame({
        'temperature': np.round(temperature, 2),
        'vibration': np.round(vibration, 2),
        'pressure': np.round(pressure, 2),
        'operating_hours': np.round(operating_hours, 1),
        'failure': failure
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sensor_data.csv', index=False)
    print("✅ Dataset generado exitosamente en 'data/sensor_data.csv'")

if __name__ == "__main__":
    generate_sensor_data()
