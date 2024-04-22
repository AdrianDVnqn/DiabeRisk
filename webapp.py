import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from streamlit.components.v1 import iframe
import base64
import joblib
import os
from streamlit_option_menu import option_menu
from streamlit_modal import Modal
from streamlit_navigation_bar import st_navbar

st.set_page_config(layout="wide")

# Crear el menu superior con los botones
botones = ['Portada', 'Dashboard', 'Predictor']

styles = {
    "nav": {
        "background-color": "#7BD192",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "padding": "0.4375rem 0.625rem",
        "margin": "0 0.125rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}
options = {
    "show_menu": False,
    "show_sidebar": False,
}
selection = st_navbar(
    botones,
    styles=styles,
    options=options,
    logo_path='diaberisk-logo.svg',
    logo_page="Portada",
    selected="Portada"
)

## Crear la barra lateral con los botones
# with st.sidebar:
#     selection=option_menu(
#         menu_title="",
#         options = ['Portada', 'Dashboard', 'Predictor'],
#         icons = ["house-heart-fill", "calendar2-heart-fill","wrench"],
#         #menu_icon = "hospital-fill",
#         default_index=0
#     )


# #Reducir el espacio en blanco al inicio de la página
# st.markdown("""
#         <style>
#                .block-container {
#                     padding-top: 2rem;
#                     padding-bottom: 0rem;
#                     padding-left: 2rem;
#                     padding-right: 2rem;
#                 }
#         </style>
#         """, unsafe_allow_html=True)

# # Inject custom CSS to set the width of the sidebar (200 px is the minimum possible)
# st.markdown(
#     """
#     <style>
#         section[data-testid="stSidebar"] {
#             width: 200px !important; # Set the width to your desired value
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# Definir los nombres completos de las variables
variable_names = {
    'time_in_hospital': 'Tiempo en el hospital',
    'num_lab_procedures': 'Número de procedimientos de laboratorio',
    'num_procedures': 'Número de procedimientos',
    'num_medications': 'Número de medicamentos',
    'number_outpatient': 'Número de visitas ambulatorias',
    'number_emergency': 'Número de emergencias',
    'number_inpatient': 'Número de hospitalizaciones',
    'number_diagnoses': 'Número de diagnósticos',
    'race': 'Etnia',
    'gender': 'Género',
    'age': 'Edad',
    'admission_type_id': 'ID del tipo de admisión',
    'discharge_disposition_id': 'ID de la disposición del alta',
    'admission_source_id': 'ID de la fuente de admisión',
    'diag_1': 'Diagnóstico 1',
    'diag_2': 'Diagnóstico 2',
    'diag_3': 'Diagnóstico 3',
    'max_glu_serum': 'Máximo suero de glucosa',
    'A1Cresult': 'Resultado de A1C',
    'metformin': 'Metformina',
    'glimepiride': 'Glimepirida',
    'glipizide': 'Glipizida',
    'glyburide': 'Gliburida',
    'pioglitazone': 'Pioglitazona',
    'rosiglitazone': 'Rosiglitazona',
    'insulin': 'Insulina',
    'change': 'Cambio',
    'diabetesMed': 'Medicamento para la diabetes'
}

#Diccionario de descripciones de las variables para ser utilizado en tooltips
descriptions = {
    'encounter_id': 'Identificador único de un encuentro',
    'patient_nbr': 'Identificador único de un paciente',
    'race': 'Etnia del paciente',
    'gender': 'Género del paciente',
    'age': 'Edad del paciente',
    'weight': 'Peso del paciente en libras',
    'admission_type_id': 'Identificador de tipo de admisión',
    'discharge_disposition_id': 'Identificador de disposición de alta',
    'admission_source_id': 'Identificador de fuente de admisión',
    'time_in_hospital': 'Número de días de estancia en el hospital',
    'payer_code': 'Código del pagador',
    'medical_specialty': 'Especialidad médica del médico que admitió al paciente',
    'num_lab_procedures': 'Número de pruebas de laboratorio realizadas durante el encuentro',
    'num_procedures': 'Número de procedimientos (que no sean pruebas de laboratorio) realizados durante el encuentro',
    'num_medications': 'Número de nombres genéricos distintos administrados durante el encuentro',
    'number_outpatient': 'Número de visitas ambulatorias del paciente en el año anterior al encuentro',
    'number_emergency': 'Número de visitas de emergencia del paciente en el año anterior al encuentro',
    'number_inpatient': 'Número de visitas hospitalarias del paciente en el año anterior al encuentro',
    'diag_1': 'Diagnóstico primario (codificado como los tres primeros dígitos de ICD9)',
    'diag_2': 'Diagnóstico secundario (codificado como los tres primeros dígitos de ICD9)',
    'diag_3': 'Diagnóstico secundario adicional (codificado como los tres primeros dígitos de ICD9)',
    'number_diagnoses': 'Número de diagnósticos ingresados en el sistema',
    'max_glu_serum': 'Indica el rango del resultado o si la prueba no se realizó',
    'A1Cresult': 'Indica el rango del resultado o si la prueba no se realizó',
    'metformin': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'repaglinide': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'nateglinide': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'chlorpropamide': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'glimepiride': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'acetohexamide': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'glipizide': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'glyburide': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'tolbutamide': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'pioglitazone': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'rosiglitazone': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'acarbose': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'miglitol': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'troglitazone': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'tolazamide': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'examide': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'citoglipton': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'insulin': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'glyburide-metformin': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'glipizide-metformin': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'glimepiride-pioglitazone': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'metformin-rosiglitazone': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'metformin-pioglitazone': 'Indica si se recetó el medicamento o hubo un cambio en la dosis',
    'change': 'Indica si hubo un cambio en los medicamentos para la diabetes',
    'diabetesMed': 'Indica si se recetó algún medicamento para la diabetes',
    'readmitted': 'Días hasta la readmisión hospitalaria. Valores: <30 si el paciente fue readmitido en menos de 30 días, >30 si el paciente fue readmitido en más de 30 días, y No si no hay registro de readmisión'
}

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

# Agrupar variables según similitud
variable_groups = {
    'Medicamentos': [
        'metformin', 'glimepiride', 'glipizide', 'glyburide',
        'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed'
    ],
    'Diagnósticos': ['diag_1', 'diag_2', 'diag_3'],
    'Información del Paciente': [
        'race', 'gender', 'age', 'admission_type_id',
        'discharge_disposition_id', 'admission_source_id'
    ],
    'Historial Médico': [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses'
    ],
    'Resultados de Pruebas': ['max_glu_serum', 'A1Cresult']
}

# Crear DataFrame para guardar los datos ingresados por el usuario
datos_prediccion = pd.DataFrame(columns=numerical_cols + categorical_cols)

# Mostrar el contenido correspondiente a la selección
if selection == "Portada":
    # #Reducir el espacio en blanco al inicio de la página
# st.markdown("""
#         <style>
#                .stAppViewBlockContainer {
#                     padding-top: 2rem;
#                     padding-bottom: 0rem;
#                     padding-left: 2rem;
#                     padding-right: 2rem;
#                 }
#         </style>
#         """, unsafe_allow_html=True)
    st.write("Bienvenido a DiabeRisk. Elija la opción Dashboard para ver datos y visualizaciones, o Predictor para acceder a la herramienta predictora del reingreso de pacientes diabéticos")

elif selection == "Dashboard":
    st.write("Aquí puedes ver datos y visualizaciones de los registros de pacientes.")

    # URL de tu dashboard de Tableau
    # Dashboard de Tableau responsive
    components.html(
    """
<div class='tableauPlaceholder' id='viz1713472472192' style='position: relative'>
    <noscript>
        <a href='#'>
            <img alt='General ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Di&#47;DiabeRisk2&#47;General&#47;1_rss.png' style='border: none' />
        </a>
    </noscript>
    <object class='tableauViz'  style='display:none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
        <param name='embed_code_version' value='3' />
        <param name='site_root' value='' />
        <param name='name' value='DiabeRisk2&#47;General' />
        <param name='tabs' value='no' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Di&#47;DiabeRisk2&#47;General&#47;1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='es-ES' />
        <param name='filter' value='publish=yes' />
    </object>
</div>
<script type='text/javascript'>
    var divElement = document.getElementById('viz1713472472192');
    var vizElement = divElement.getElementsByTagName('object')[0];
    if ( divElement.offsetWidth > 800 ) {
        vizElement.style.width='100%';
        vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
    } else if ( divElement.offsetWidth > 500 ) {
        vizElement.style.width='100%';
        vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
    } else {
        vizElement.style.width='100%';
        vizElement.style.height='1777px';
    }
    var scriptElement = document.createElement('script');
    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
    vizElement.parentNode.insertBefore(scriptElement, vizElement);
</script>
    """,
    height=830,
    width=1500,
    )

    # Mostrar la página Predictor
else:

    # Cargar el DataFrame preprocesado
    try:
        df = pd.read_csv('data.csv')
        os.write(1,b'DataFrame cargado correctamente.\n')
    except Exception as e:
        st.caption(f"Error al cargar el DataFrame: {e}")
    # Cargar el modelo de machine learning previamente entrenado
    try:
        model = joblib.load('pipeline_xgb.pkl')
        os.write(1,b'Modelo de ML cargado correctamente.\n')
    except Exception as e:
        st.caption(f"Error al cargar el modelo: {e}")

    def realizar_prediccion(datos_prediccion, model):
      try:
          # Verificar si todos los datos necesarios están presentes
          #if any(pd.isnull(datos_prediccion[numerical_cols])):
          #  raise ValueError("Es necesario cargar todos los datos para realizar la predicción")
          # Realizar la predicción
          prediccion = model.predict(datos_prediccion)[0]
          probabilidad = model.predict_proba(datos_prediccion)[0][1]  # Probabilidad de la clase positiva (readmisión)

          # Mostrar el resultado de la predicción y la probabilidad
          if prediccion == 1:
            with popup.container():
                st.markdown(f"🩺**Predicción: 🚨Hay readmisión del paciente.** Probabilidad: **{probabilidad*100:.1f}%**")
          else:
            with popup.container():
                st.markdown(f"🩺**Predicción:** 🟢No hay readmisión del paciente. Probabilidad: **{probabilidad*100:.1f}%**")
      except Exception as e:
            with popup.container():
                st.markdown(f'⚠️Error al realizar la predicción: {e}')

    st.write("Ingrese a continuación los datos del paciente. Para obtener más información sobre cada campo, coloque el cursor sobre el símbolo de pregunta (?).")

    # Dividir la página en columnas
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    # Cajas de ingreso de datos para variables numéricas y categóricas
    for group, cols in variable_groups.items():
        if group == 'Medicamentos':
            col_group = col1
        elif group == 'Diagnósticos':
            col_group = col2
        elif group == 'Información del Paciente':
            col_group = col3
        elif group == 'Historial Médico':
            col_group = col4
        elif group == 'Resultados de Pruebas':
            col_group = col5

        with col_group:
            st.write(f"**{group}**")
            for col_name in cols:
                if col_name in numerical_cols:
                    value = st.number_input(variable_names[col_name], step=1, help=descriptions.get(col_name, "Sin descripción"), min_value=0)
                    datos_prediccion[col_name] = [value]
                elif col_name in categorical_cols:
                    values = df[col_name].unique()
                    value = st.selectbox(variable_names[col_name], values, help=descriptions.get(col_name, "Sin descripción"))
                    datos_prediccion[col_name] = [value]

    # Mostrar el DataFrame de datos del paciente y el botón de predicción

    #st.write("Datos del paciente:") #Creo que para el deploy ya no vale la pena mostrar esto
    #st.dataframe(datos_prediccion)

    popup = Modal(key="results", title="Resultados")

    # Botón para realizar la predicción
    if st.button("Realizar Predicción"):
            #popup.open()
            realizar_prediccion(datos_prediccion, model)
            
            
