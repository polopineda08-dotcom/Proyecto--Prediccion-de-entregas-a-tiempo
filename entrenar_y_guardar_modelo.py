import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Cargar datos
df = pd.read_csv("Entregas.csv")

# Preparar datos
X = df.drop(['Reached.on.Time_Y.N', 'ID'], axis=1)
y = df['Reached.on.Time_Y.N']
X = pd.get_dummies(X)

# Entrenar modelo
modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
modelo.fit(X, y)

# Guardar modelo
joblib.dump(modelo, "modelo_entregas.joblib")
joblib.dump(X.columns.tolist(), "columnas_modelo.joblib")

print("✅ Modelo guardado correctamente")
