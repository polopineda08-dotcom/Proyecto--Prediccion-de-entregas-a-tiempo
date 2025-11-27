# ==========================================
# MODELO DE MACHINE LEARNING
# Proyecto: Predicción de Entregas a Tiempo
# Archivo: modelo_entregas.py
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Cargar los datos
df = pd.read_csv("Entregas.csv")

# 2. Columnas categóricas
columnas_categoricas = [
    'Warehouse_block',
    'Mode_of_Shipment',
    'Product_importance',
    'Gender'
]

# 3. Convertir texto a números
le = LabelEncoder()
for col in columnas_categoricas:
    df[col] = le.fit_transform(df[col])

# 4. Separar variables de entrada (X) y objetivo (y)
X = X = df.drop(['Reached.on.Time_Y.N', 'ID'], axis=1)
y = df['Reached.on.Time_Y.N']

# 5. Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 6. Crear y entrenar el modelo
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# 7. Hacer predicciones
y_pred = modelo.predict(X_test)

# 8. Resultados del modelo
print("\n===== RESULTADOS DEL MODELO =====")

print("\nAccuracy del modelo:")
print(accuracy_score(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

print("\n===== ENTRENANDO MODELO MEJORADO: RANDOM FOREST =====")

# Crear el modelo Random Forest
modelo_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
# Entrenar el modelo
modelo_rf.fit(X_train, y_train)

# Hacer predicciones
y_pred_rf = modelo_rf.predict(X_test)

# Evaluar el modelo mejorado
print("\n===== RESULTADOS DEL RANDOM FOREST =====")

print("\nAccuracy del modelo:")
print(accuracy_score(y_test, y_pred_rf))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_rf))
from sklearn.model_selection import GridSearchCV

print("\n===== OPTIMIZACIÓN DE RANDOM FOREST =====")

# Parámetros que vamos a probar
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)

# GridSearch para buscar mejores parámetros
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy'
)

# Entrenar búsqueda
grid_search.fit(X_train, y_train)

print("Mejores parámetros encontrados:")
print(grid_search.best_params_)

# Usar el mejor modelo encontrado
mejor_modelo = grid_search.best_estimator_

# Predicciones
y_pred_opt = mejor_modelo.predict(X_test)

# Resultados
print("\n===== RESULTADOS DEL MODELO OPTIMIZADO =====")
print("Accuracy:")
print(accuracy_score(y_test, y_pred_opt))
