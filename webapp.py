
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from streamlit.components.v1 import iframe
import base64
import joblib

# Definir las variables numéricas y categóricas
numerical_cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]
categorical_cols = [
    'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult',
    'metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed'
]

# Crear DataFrame para guardar los datos ingresados por el usuario
datos_prediccion = pd.DataFrame(columns=numerical_cols + categorical_cols)

def get_base64(file_path):
    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode('utf-8')
    return encoded_string
    
def set_background(png_file, opacity):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    opacity: {opacity};
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("hosp.jpg", 0.5)

#Pongo la fuente en negro
st.markdown("<style>body {color: black;}</style>", unsafe_allow_html=True)
st.markdown("<style>h1 {color: black;}</style>", unsafe_allow_html=True)
st.markdown("<style>h2 {color: black;}</style>", unsafe_allow_html=True)
st.markdown("<style>h3 {color: black;}</style>", unsafe_allow_html=True)
st.markdown("<style>h4 {color: black;}</style>", unsafe_allow_html=True)
st.markdown("<style>h5 {color: black;}</style>", unsafe_allow_html=True)
st.markdown("<style>h6 {color: black;}</style>", unsafe_allow_html=True)
st.markdown("<style>div.Widget.row-widget.stRadio > div{color: black;}</style>", unsafe_allow_html=True)
st.markdown("<style>.css-1v3fvcr {color: black;}</style>", unsafe_allow_html=True)
st.markdown("<style>.css-j7fq0y {color: black;}</style>", unsafe_allow_html=True)

# Crear la barra lateral con los botones
selection = st.sidebar.radio("Seleccionar página:", ("Portada", "Dashboard", "Predictor"))

# Mostrar el contenido correspondiente a la selección
if selection == "Portada":
    st.image("diaberisk.png")
    st.write("Bienvenido a DiabeRisk. Elija la opción Dashboard para ver datos y visualizaciones, o Predictor para acceder a la herramienta predictora del reingreso de pacientes diabéticos")

elif selection == "Dashboard":
    st.title("Dashboard")
    st.write("Aquí puedes ver datos y visualizaciones de los registros de pacientes.")

    # URL de tu dashboard de Tableau
    components.html(
    """
    <div class='tableauPlaceholder' id='viz1713308822056' style='position: relative'>
        <noscript>
            <a href='#'>
                <img alt='Principal' src='https://public.tableau.com/static/images/Di/DiabeRisk/Principal/1_rss.png' style='border: none' />
            </a>
        </noscript>
        <object class='tableauViz'  style='display:none;'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
            <param name='embed_code_version' value='3' />
            <param name='site_root' value='' />
            <param name='name' value='DiabeRisk/Principal' />
            <param name='tabs' value='no' />
            <param name='toolbar' value='yes' />
            <param name='static_image' value='https://public.tableau.com/static/images/Di/DiabeRisk/Principal/1.png' />
            <param name='animate_transition' value='yes' />
            <param name='display_static_image' value='yes' />
            <param name='display_spinner' value='yes' />
            <param name='display_overlay' value='yes' />
            <param name='display_count' value='yes' />
            <param name='language' value='es-ES' />
        </object>
    </div>
    <script type='text/javascript'>
        var divElement = document.getElementById('viz1713308822056');
        var vizElement = divElement.getElementsByTagName('object')[0];
        if ( divElement.offsetWidth > 800 ) {
            vizElement.style.width='1000px';
            vizElement.style.height='827px';
        } else if ( divElement.offsetWidth > 500 ) {
            vizElement.style.width='1000px';
            vizElement.style.height='827px';
        } else {
            vizElement.style.width='100%';
            vizElement.style.height='727px';
        }
        var scriptElement = document.createElement('script');
        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
        vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """,
    height=900,
    width=1000,
    )


# Mostrar las cajas de ingreso de datos
else:
    st.title("Predictor")
    st.write("Aquí puedes usar el predictor.")
    # Cargar el DataFrame preprocesado
    try:
        df = pd.read_csv('data.csv')
        st.write("DataFrame cargado correctamente")
    except Exception as e:
        st.write(f"Error al cargar el DataFrame: {e}")
    # Cargar el modelo de machine learning previamente entrenado
    try:
        model = joblib.load('pipeline_xgb.pkl')
        st.write("Modelo cargado correctamente")
    except Exception as e:
        st.write(f"Error al cargar el modelo: {e}")

    # Cajas de ingreso de datos para variables numéricas
    for col in numerical_cols:
        value = st.text_input(col, f"Ingrese el valor de {col}")
        datos_prediccion[col] = [value]

    # Cajas de ingreso de datos para variables categóricas
    for col in categorical_cols:
        values = df[col].unique()
        value = st.selectbox(col, values)
        datos_prediccion[col] = [value]

    st.write("Datos ingresados:")
    st.write(datos_prediccion)

    def realizar_prediccion(datos_prediccion, model):
        try:
            # Realizar la predicción
            prediccion = model.predict(datos_prediccion)[0]
            probabilidad = model.predict_proba(datos_prediccion)[0][1]  # Probabilidad de la clase positiva (readmisión)
        
            # Mostrar el resultado de la predicción y la probabilidad
            if prediccion == 1:
                st.write(f"Predicción: Hay readmisión del paciente. Probabilidad: {probabilidad:.2f}")
            else:
                st.write(f"Predicción: No hay readmisión del paciente. Probabilidad: {probabilidad:.2f}")
        except Exception as e:
            st.write(f'Error al realizar la predicción: {e}')


    # Botón para realizar la predicción
    if st.button("Realizar Predicción"):
        realizar_prediccion(datos_prediccion, model)

