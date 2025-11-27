# ==========================================
# PROYECTO: PREDICCIÓN DE ENTREGAS A TIEMPO
# ARCHIVO: eda_entregas.py
# ==========================================

# ========= 1. IMPORTAR LIBRERÍAS ==========
import pandas as pd
import matplotlib.pyplot as plt

# Configuración opcional de gráficos
plt.rcParams["figure.figsize"] = (8, 5)

# ========= 2. CARGAR DATASET ==========
archivo = "Entregas.csv"
df = pd.read_csv(archivo)

# ========= 3. REVISIÓN GENERAL ==========
print("\n===== DIMENSIONES DEL DATASET =====")
print(df.shape)

print("\n===== PRIMERAS FILAS =====")
print(df.head())

print("\n===== NOMBRES DE COLUMNAS =====")
print(df.columns.tolist())

print("\n===== TIPOS DE DATOS =====")
print(df.dtypes)

print("\n===== VALORES NULOS POR COLUMNA =====")
print(df.isnull().sum())

print("\n===== DESCRIPCIÓN ESTADÍSTICA =====")
print(df.describe())

# ========= 4. LIMPIEZA DE DATOS ==========
print("\n===== LIMPIEZA DE DATOS =====")

# Revisar duplicados
duplicados = df.duplicated().sum()
print(f"Filas duplicadas encontradas: {duplicados}")

# Eliminar duplicados si existen
if duplicados > 0:
    df = df.drop_duplicates()
    print("Duplicados eliminados")

# Convertir columnas categóricas
columnas_categoricas = [
    'Warehouse_block',
    'Mode_of_Shipment',
    'Product_importance',
    'Gender'
]

for col in columnas_categoricas:
    df[col] = df[col].astype('category')

print("\nTipos de datos después de limpieza:")
print(df.dtypes)

print("\nDimensiones finales después de limpieza:")
print(df.shape)

# ========= 5. EDA - VARIABLE OBJETIVO ==========
print("\n===== DISTRIBUCIÓN DE LA VARIABLE OBJETIVO =====")
print(df['Reached.on.Time_Y.N'].value_counts())

plt.figure()
df['Reached.on.Time_Y.N'].value_counts().plot(kind='bar')
plt.title("Entregas a tiempo (0) vs tarde (1)")
plt.xlabel("Estado de entrega")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# ========= 6. EDA - VARIABLES CATEGÓRICAS ==========
print("\n===== VARIABLES CATEGÓRICAS =====")
cat_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']

for col in cat_cols:
    print(f"\n--- Distribución de {col} ---")
    print(df[col].value_counts())

    plt.figure()
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Distribución de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

# ========= 7. EDA - VARIABLES NUMÉRICAS ==========
print("\n===== VARIABLES NUMÉRICAS =====")

num_cols = [
    'Customer_care_calls',
    'Customer_rating',
    'Cost_of_the_Product',
    'Prior_purchases',
    'Discount_offered',
    'Weight_in_gms'
]

for col in num_cols:
    plt.figure()
    df[col].hist(bins=20)
    plt.title(f"Histograma de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

# ========= 8. ANÁLISIS DE RELACIONES ==========
print("\n===== PROMEDIOS POR TIPO DE ENTREGA =====")
print(df.groupby('Reached.on.Time_Y.N')[num_cols].mean())

# ========= 9. MATRIZ DE CORRELACIÓN ==========
print("\n===== MATRIZ DE CORRELACIÓN =====")
corr = df[num_cols + ['Reached.on.Time_Y.N']].corr()
print(corr)

plt.figure()
plt.imshow(corr, interpolation='nearest', cmap='viridis')
plt.colorbar(label='Correlación')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Mapa de calor de correlación")
plt.tight_layout()
plt.show()

# ========= 9.5 ANÁLISIS DE DATOS FALTANTES Y ATÍPICOS ==========
print("\n===== ANÁLISIS DE DATOS FALTANTES =====")

# Valores nulos
nulos = df.isnull().sum()
print(nulos)

print("\nColumnas con valores nulos:")
print(nulos[nulos > 0])

print("\n===== DETECCIÓN DE DATOS ATÍPICOS (OUTLIERS) =====")

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Límites para detectar outliers
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Filtrar outliers
    outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]

    # Imprimir resultados
    print(f"\nColumna: {col}")
    print(f"Límite inferior: {limite_inferior}")
    print(f"Límite superior: {limite_superior}")
    print(f"Número de valores atípicos: {len(outliers)}")

    # ===== BOXPLOT CON COLORES =====
    plt.figure(figsize=(8, 4))

    plt.boxplot(
        df[col],
        vert=False,
        patch_artist=True,  # Permite colorear la caja
        boxprops=dict(facecolor='skyblue', color='blue'),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markerfacecolor='orange', markersize=6, linestyle='none')
    )

    plt.title(f"Boxplot - Detección de Outliers en {col}", fontsize=12)
    plt.xlabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# ========= 10. FIN DEL EDA ==========
print("\n===== EDA COMPLETADO CORRECTAMENTE =====")
