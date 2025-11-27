# ==========================================
# APP STREAMLIT - PREDICCIÓN DE ENTREGAS
# ==========================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# CONFIGURACIÓN BÁSICA
# -------------------------------
st.set_page_config(
    page_title="Predicción de Entregas a Tiempo",
    layout="wide"
)

# -------------------------------
# CARGA DE DATOS
# -------------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv("Entregas.csv")

df = cargar_datos()

# -------------------------------
# CARGAR MODELO REAL
# -------------------------------
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_entregas.joblib")
    columnas = joblib.load("columnas_modelo.joblib")
    return modelo, columnas

modelo, columnas_modelo = cargar_modelo()

# -------------------------------
# PREPARAR DATA
# -------------------------------
X = df.drop(["Reached.on.Time_Y.N", "ID"], axis=1)
y = df["Reached.on.Time_Y.N"]
X = pd.get_dummies(X)
X = X.reindex(columns=columnas_modelo, fill_value=0)

# -------------------------------
# MÉTRICAS REALES
# -------------------------------
y_pred = modelo.predict(X)
acc = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)
cr = classification_report(y, y_pred)

# -------------------------------
# MENÚ
# -------------------------------
menu = st.sidebar.radio(
    "Navegación",
    ["Inicio", "Visualizaciones", "Dashboard", "Modelo ML", "Preguntas de negocio"]
)

# ======================================================
# 1) INICIO
# ======================================================
if menu == "Inicio":
    st.title("📦 Predicción de Entregas a Tiempo")

    st.markdown("""
    ### Descripción del problema

    La empresa desea **predecir si una entrega llegará a tiempo o con retraso**, 
    utilizando información logística como:
    - Peso del producto  
    - Descuento ofrecido  
    - Llamadas a servicio al cliente  
    - Tipo de envío, almacén, etc.  

    Esta app muestra un dashboard, visualizaciones y un modelo de Machine Learning 
    entrenado para apoyar la toma de decisiones en logística.
    """)

    total = len(df)
    a_tiempo = (df["Reached.on.Time_Y.N"] == 0).sum()
    tarde = (df["Reached.on.Time_Y.N"] == 1).sum()
    porc_tarde = (tarde / total) * 100
    porc_atiempo = (a_tiempo / total) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de envíos", total)
    col2.metric("Entregas a tiempo (0)", a_tiempo, f"{porc_atiempo:.1f}%")
    col3.metric("Entregas tarde (1)", tarde, f"{porc_tarde:.1f}%")

    st.subheader("Muestra del dataset")
    st.dataframe(df.head())

# ======================================================
# 2) VISUALIZACIONES
# ======================================================
elif menu == "Visualizaciones":
    st.title("📊 Visualizaciones principales")

    # 1. Barras: entregas a tiempo vs tarde
    st.subheader("1️⃣ Distribución de entregas (a tiempo vs tarde)")
    st.markdown("""
    **¿Qué muestra esta gráfica?**  
    Compara cuántos pedidos llegaron **a tiempo** y cuántos llegaron **tarde**. 

    **¿Para qué sirve?**  
    Nos permite ver rápidamente si la empresa tiene más problemas con retrasos o si la mayoría de las entregas se realizan correctamente.
    """)

    fig1, ax1 = plt.subplots()
    df["Reached.on.Time_Y.N"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_xticklabels(["A tiempo (0)", "Tarde (1)"], rotation=0)
    ax1.set_ylabel("Cantidad de envíos")
    st.pyplot(fig1)

    st.markdown("""
    ### ✅ Conclusión
                
    Una **alta cantidad de entregas tardías** indica oportunidades claras de mejora en los procesos logísticos.  
    Este resultado muestra que la empresa **debe optimizar rutas, control de inventarios y tiempos de despacho** para mejorar la satisfacción del cliente.
    """)

    # 2. Histograma del costo del producto
    st.subheader("2️⃣ Histograma del costo del producto")
    st.markdown("""
    **¿Qué muestra esta gráfica?**  
    Representa la distribución de precios de los productos.

    **¿Para qué sirve?**  
    Ayuda a identificar si los productos más caros tienden a ser menos frecuentes y cómo se distribuyen los precios.
    """)

    fig2, ax2 = plt.subplots()

    ax2.hist(
    df["Cost_of_the_Product"],
    bins=20,             # Número de barras
    edgecolor="black",   # Bordes en negro
    rwidth=0.9           # Separación entre barras (más chico = más espacio)
)

    ax2.set_xlabel("Costo del producto")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Distribución del costo de los productos")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    st.pyplot(fig2)

    st.markdown("""
     ### ✅ Conclusión
     Los productos con **mayor costo pueden representar un mayor riesgo operativo**, ya que suelen requerir mayor control y prioridad en la entrega.  
     Detectar estos patrones ayuda a la empresa a **reducir pérdidas económicas y mejorar la experiencia del cliente.**
     """)

    # 3. Boxplot: Peso vs entrega
    st.subheader("3️⃣ Relación entre peso y estado de entrega")
    st.markdown("""
    **¿Qué muestra esta gráfica?**  
    Compara el peso de los paquetes según si llegaron a tiempo o con retraso.

    **¿Para qué sirve?**  
    Permite identificar si los paquetes más pesados tienen mayor tendencia a llegar tarde.
    """)
    fig3, ax3 = plt.subplots()
    df.boxplot(column="Weight_in_gms", by="Reached.on.Time_Y.N", ax=ax3)
    ax3.set_title("Peso por tipo de entrega")
    ax3.set_xlabel("Entrega (0=A tiempo, 1=Tarde)")
    ax3.set_ylabel("Peso (g)")
    st.pyplot(fig3)
    st.markdown("""
     ### ✅ Conclusión

     Los paquetes más pesados tienden a presentar **mayor probabilidad de retraso**, lo que evidencia limitaciones en la capacidad logística y de transporte.  
     Esta información es clave para **optimizar la asignación de recursos y mejorar los tiempos de entrega.**
     """)


# ======================================================
# 3) DASHBOARD (KPIs + FILTROS)
# ======================================================
elif menu == "Dashboard":
    st.title("📈 Dashboard logístico")

    st.markdown("Filtra los datos para analizar segmentos específicos.")

    col_f1, col_f2 = st.columns(2)

    # Filtro por tipo de envío
    tipos_envio = ["Todos"] + sorted(df["Mode_of_Shipment"].unique().tolist())
    filtro_envio = col_f1.selectbox("Tipo de envío", tipos_envio)

    # Filtro por bloque de almacén
    bloques = ["Todos"] + sorted(df["Warehouse_block"].unique().tolist())
    filtro_bloque = col_f2.selectbox("Bloque de almacén", bloques)

    df_filtrado = df.copy()

    if filtro_envio != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Mode_of_Shipment"] == filtro_envio]

    if filtro_bloque != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Warehouse_block"] == filtro_bloque]

    st.subheader("KPIs del segmento filtrado")

    total_f = len(df_filtrado)
    a_tiempo_f = (df_filtrado["Reached.on.Time_Y.N"] == 0).sum()
    tarde_f = (df_filtrado["Reached.on.Time_Y.N"] == 1).sum()

    if total_f > 0:
        porc_tarde_f = (tarde_f / total_f) * 100
        porc_atiempo_f = (a_tiempo_f / total_f) * 100
    else:
        porc_tarde_f = porc_atiempo_f = 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de envíos (filtro)", total_f)
    c2.metric("A tiempo (0)", a_tiempo_f, f"{porc_atiempo_f:.1f}%")
    c3.metric("Tarde (1)", tarde_f, f"{porc_tarde_f:.1f}%")

    st.subheader("Distribución de entregas en el segmento")
    if total_f > 0:
        fig4, ax4 = plt.subplots()
        df_filtrado["Reached.on.Time_Y.N"].value_counts().plot(kind="bar", ax=ax4)
        ax4.set_xticklabels(["A tiempo (0)", "Tarde (1)"], rotation=0)
        st.pyplot(fig4)
    else:
        st.info("No hay datos para el filtro seleccionado.")

# ======================================================
# 4) MODELO DE MACHINE LEARNING (MEJORADO)
# ======================================================
elif menu == "Modelo ML":
    st.title("🤖 Modelo ML en Producción")

    st.markdown("""
    Este modelo fue previamente entrenado y cargado desde archivo.
    Use los controles para simular un nuevo escenario logístico.
    """)

    # ======================
    # BLOQUE 1 – MÉTRICAS
    # ======================
    st.subheader("📊 Desempeño del Modelo")

    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("Accuracy", f"{acc:.2f}")
    colm2.metric("Total clases", "2")
    colm3.metric("Registros evaluados", len(df))

    with st.expander("Ver matriz de confusión"):
        st.write(cm)

    with st.expander("Ver reporte de clasificación"):

      # Convertir el reporte a diccionario
      cr_dict = classification_report(y, y_pred, output_dict=True)

      # Convertir a DataFrame
      cr_df = pd.DataFrame(cr_dict).transpose()

      # Redondear a 2 decimales
      cr_df = cr_df.round(2)

      # Mostrar tabla bonita
      st.dataframe(cr_df, use_container_width=True)

    # ======================
    # BLOQUE 2 – SIMULADOR
    # ======================
    st.subheader("🎛️ Simulador de escenarios")

    col1, col2 = st.columns(2)

    with col1:
        customer_calls = st.slider("📞 Llamadas a soporte", 0, 10, 3)
        customer_rating = st.slider("⭐ Calificación del cliente", 1, 5, 3)
        prior_purchases = st.slider("🛒 Compras previas", 0, 10, 3)

    with col2:
        cost = st.slider("💲 Costo del producto", 50, 500, 200)
        discount = st.slider("🏷️ Descuento (%)", 0, 70, 10)
        weight = st.slider("⚖️ Peso (g)", 100, 8000, 2000)

    # Crear datos de entrada
    input_data = pd.DataFrame([{
        "Customer_care_calls": customer_calls,
        "Customer_rating": customer_rating,
        "Prior_purchases": prior_purchases,
        "Cost_of_the_Product": cost,
        "Discount_offered": discount,
        "Weight_in_gms": weight
    }])

    input_data = input_data.reindex(columns=columnas_modelo, fill_value=0)

    st.divider()

    # ======================
    # BLOQUE 3 – RESULTADO
    # ======================
    if st.button("🔍 Predecir resultado", key="btn_prediccion"):
        resultado = modelo.predict(input_data)[0]
        probas = modelo.predict_proba(input_data)[0]

        st.subheader("📌 Resultado de la predicción")

        if resultado == 0:
            st.success(f"✅ La entrega llegaría **A TIEMPO**")
        else:
            st.error(f"⚠️ La entrega llegaría **TARDE**")

        st.write("### Probabilidades del modelo")
        st.progress(float(probas[0]))
        st.caption(f"A tiempo: {probas[0]*100:.1f}%")

        st.progress(float(probas[1]))
        st.caption(f"Tarde: {probas[1]*100:.1f}%")

        
# ======================================================
# 5) PREGUNTAS DE NEGOCIO
# ======================================================
elif menu == "Preguntas de negocio":
    st.title("📌 Preguntas de negocio")

    st.markdown("""
    Estas visualizaciones responden a las preguntas clave del proyecto:

    1️⃣ ¿Qué factores influyen más en que una entrega llegue tarde?  
    2️⃣ ¿Qué tipo de envío tiene mayor probabilidad de retrasos?  
    3️⃣ ¿Qué variables operativas afectan más el desempeño logístico?
    """)

    # 1️⃣ Factores que más influyen (importancia de variables)
    st.subheader("1️⃣ Factores que más influyen en los retrasos")

    importances = modelo.feature_importances_
    importancia_df = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": importances
    }).sort_values(by="Importancia", ascending=False)

    st.write("Top 10 variables más importantes según el modelo:")
    st.dataframe(importancia_df.head(10))

    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(importancia_df["Variable"][:10], importancia_df["Importancia"][:10])
    ax_imp.invert_yaxis()
    ax_imp.set_title("Top 10 variables más importantes")
    st.pyplot(fig_imp)

    # 2️⃣ Tipo de envío con más retrasos
    st.subheader("2️⃣ Tipo de envío con mayor probabilidad de retrasos")

    fig_envio, ax_envio = plt.subplots()
    df.groupby("Mode_of_Shipment")["Reached.on.Time_Y.N"].mean().plot(kind="bar", ax=ax_envio)
    ax_envio.set_ylabel("Proporción de entregas tarde")
    ax_envio.set_title("Retrasos promedio por tipo de envío")
    st.pyplot(fig_envio)

    # 3️⃣ Impacto del peso en el desempeño logístico
    st.subheader("3️⃣ Impacto del peso en el desempeño logístico")

    fig_box, ax_box = plt.subplots()
    df.boxplot(column="Weight_in_gms", by="Reached.on.Time_Y.N", ax=ax_box)
    ax_box.set_title("Peso por tipo de entrega (0=A tiempo, 1=Tarde)")
    ax_box.set_ylabel("Peso (g)")
    st.pyplot(fig_box)

    st.markdown("""
    **Conclusión de negocio:**  
    El análisis muestra que variables como el **peso del producto**, el **descuento ofrecido** 
    y características operativas específicas tienen un impacto importante en los retrasos. 
    Además, ciertos tipos de envío presentan una mayor proporción de entregas tardías, 
    lo que puede guiar decisiones de mejora en la logística.
    """)

