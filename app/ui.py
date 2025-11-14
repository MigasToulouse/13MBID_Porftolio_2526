import streamlit as st
import requests
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Predicción de suscripción bancaria",
    page_icon=":bar_chart:",
    layout="wide"
)

# Título y descripción
st.title("Predicción de suscripción a un producto bancario")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning para predecir si un cliente 
suscribirá un depósito a plazo fijo basado en sus características personales y de contacto.
""")

# URL de la API
API_URL = st.sidebar.text_input("URL de la API", "http://localhost:8000")

# Verificar estado de la API
st.sidebar.markdown("---")
st.sidebar.header("Estado de la API")

try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    if health_response.status_code == 200:
        health_data = health_response.json()
        st.sidebar.success("API está en línea :green_circle:")
    else:
        st.sidebar.error("API no está disponible :red_circle:")
except Exception as e:
    st.sidebar.error(f"Error al conectar con la API: {e}")

# Crear pestañas
tab1, tab2 = st.tabs(["Predicción Individual", "Información del Modelo"])

with tab1:
    st.header("Información del Cliente")

    # Crear columnas para organizar los campos
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Datos Personales")
        age = st.number_input("Edad", min_value=18, max_value=100, value=30)
        job = st.selectbox("Tipo de trabajo", ["admin.", "blue-collar", "entrepreneur", "housemaid", 
                                               "management", "retired", "self-employed", "services", 
                                               "student", "technician", "unemployed", "unknown"])
        marital = st.selectbox("Estado civil", ["divorced", "married", "single", "unknown"])

        education = st.selectbox("Nivel educativo", ["basic.4y", "basic.6y", "basic.9y", 
                                                     "high.school", "illiterate", "professional.course", 
                                                     "university.degree", "unknown"])
        
        housing = st.selectbox("¿Tiene hipoteca? 0 - no, 1 - si", [0, 1])
        loan = st.selectbox("¿Tiene préstamo personal? 0 - no, 1 - si", [0, 1])

    with col2:
        st.subheader("Datos de Contacto")
        contact = st.selectbox("Tipo de contacto", ["cellular", "telephone"])
        month = st.selectbox("Mes de contacto", ["jan", "feb", "mar", "apr", "may", "jun", 
                                                 "jul", "aug", "sep", "oct", "nov", "dec"])
        day_of_week = st.selectbox("Día de la semana del contacto", ["mon", "tue", "wed", 
                                                                     "thu", "fri"])
        duration = st.number_input("Duración de la llamada (segundos)", min_value=0, value=100)
        campaign = st.number_input("Número de contactos en esta campaña", min_value=0, value=1)
        previous = st.number_input("Número de contactos en campañas anteriores", min_value=0, value=0)
        poutcome = st.selectbox("Resultado de la campaña anterior", ["failure", "nonexistent", "success"])
        contacted_before = st.selectbox("¿Fue contactado antes? 0 - no, 1 - si", [0, 1])
        contacts_diff = st.number_input("Diferencia en número de contactos (actual - anterior)", value=0)

    with col3:
        st.subheader("Indicadores Económicos")
        emp_var_rate = st.number_input("Tasa de variación del empleo", value=1.1, format="%.2f")
        cons_price_idx = st.number_input("Índice de precios al consumidor", value=93.994, format="%.2f")
        cons_conf_idx = st.number_input("Índice de confianza del consumidor", value=-36.4, format="%.2f")
        euribor3m = st.number_input("Tasa Euribor a 3 meses", value=4.857, format="%.2f")
        nr_employed = st.number_input("Número de empleados", value=5191)

    st.markdown("---")
    if st.button("Realizar predicción", type = "primary", use_container_width=True):
        payload = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day_of_week": day_of_week,
            "duration": duration,
            "campaign": campaign,
            "previous": previous,
            "poutcome": poutcome,
            "contacted_before": contacted_before,
            "contacts_diff": contacts_diff,
            "emp_var_rate": emp_var_rate,
            "cons_price_idx": cons_price_idx,
            "cons_conf_idx": cons_conf_idx,
            "euribor3m": euribor3m,
            "nr_employed": nr_employed
        }

        try:
            #Realizar la petición a la API
            with st.spinner("Consultando el modelo..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

                if response.status_code == 200:
                    result = response.json()

                    # Mostrar los resultados
                    st.success("Predicción realizada con éxito!")

                    # Crear columnas para mostrar los resultados
                    res_col1, res_col2 = st.columns(2)

                    with res_col1:
                        st.markdown("### Resultado de la Predicción")
                        prediction = "si" if result["prediction"] == 1 else "no"
                        if prediction == "si":
                            st.success(f"El cliente probablemente suscribirá el depósito a plazo fijo: **{prediction}**")
                        else:
                            st.error(f"El cliente probablemente no suscribirá el depósito a plazo fijo: **{prediction}**")

                        # Información del modelo
                        st.markdown("### Información del Modelo")
                        st.json(result["model_info"])

                    with res_col2:
                        st.markdown("### Probabilidades")

                        # Crear gráfico de probabilidades
                        probabilities = result["probability"]

                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(probabilities.keys()),
                                y=list(probabilities.values()),
                                text=[f"{v*100:.2f}%" for v in probabilities.values()],
                                textposition='auto',
                                marker_color=['#EF553B' if k == "no" else '#00CC96' for k in probabilities.keys()]
                            )
                        ])

                        fig.update_layout(
                            title = "Probabilidades de suscripción",
                            yaxis_title="Probabilidad",
                            xaxis_title="Clase",
                            yaxis = dict(tickformat=".0%"),
                            height=300,
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Mostrar porbabilidades en formato de métricas
                        prob_col1, prob_col2 = st.columns(2)
                        with prob_col1:
                            st.metric(label="Probabilidad de no suscribir", value=f"{probabilities['no']*100:.2f}%")
                        with prob_col2:
                            st.metric(label="Probabilidad de suscribir", value=f"{probabilities['si']*100:.2f}%")
                    
                    #Mostrar los datos enviados
                    with st.expander("Ver datos enviados a la API"):
                        st.json(payload)

                    #Mostrar la respuesta completa de la API
                    with st.expander("Ver respuesta completa de la API"):
                        st.json(result)
        
                else:
                    st.error(f"Error en la predicción: {response.status_code} - {response.text}")
                    st.json(response.json())

        except requests.exceptions.ConnectionError:
            st.error("No se pudo conectar con la API. Verifique la URL y que la API esté en funcionamiento.")
        except requests.exceptions.Timeout:
            st.error("La solicitud a la API ha excedido el tiempo de espera.")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado: {e}")
        
with tab2:
    st.header("Información del Modelo de Machine Learning")
    st.markdown("""
    ### Características del modelo:
                
    Este modelo de Machine Learning está diseñado para predecir si un cliente bancario
    suscribirá un depósito a plazo fijo basándos en:

    ### Variables de entrada:
                
    **Datos personales:**
    - Edad
    - Tipo de trabajo
    - Estado civil
    - Nivel educativo
    - Hipoteca
    - Préstamo personal
                
    **Datos de camapaña:**
    - Tipo de contacto
    - Mes del contacto
    - Día de la semana del contacto
    - Duración de la llamada
    - Número de contactos en esta campaña
    - Número de contactos en campañas anteriores
    - Resultado de la campaña anterior
    - Si fue contactado antes

    **Indicadores económicos:**
    - Tasa de variación del empleo
    - Índice de precios al consumidor
    - Índice de confianza del consumidor
    - Tasa Euribor a 3 meses
    - Número de empleados
                
    ### Salida del modelo:
    - **Predicción**: "yes" o "no" indicando si el cliente suscribirá el depósito.
    - **Probabilidades**: Probabilidades asociadas a cada clase de salida.

    ### Tecnologías usadas:
    - **backend**: FastAPI
    - **frontend**: Streamlit
    - **modelo de ML**: Scikit-learn (Decision Tree Classifier)
    - **Preprocesamiento**: pipeline de scikit-learn.
    """)

    # Intentar obtener información adicional de la API
    try:
        root_response = requests.get(f"{API_URL}/", timeout=2)
        if root_response.status_code == 200:
            st.markdown("### Endpoints disponibles:")
            st.json(root_response.json())
    except Exception:
        pass

    st.markdown("---")
    st.info("""
            **Tip**: Para mejores predicciones, asegúrate de proporcionar datos precisos y completos del cliente. 
            La duración de la llamada y el resultado de campañas anteriores son especialmente influyentes en la predicción.)
            """)
