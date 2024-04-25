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
import shap
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import io

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


def set_background(png_file):
    with open(png_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode('utf-8')

    # Definir el estilo CSS para la imagen de fondo
    page_bg_img = f'''
    <style>
    [data-testid="stApp"] {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;

    }}
    </style>
    '''

    # Aplicar el estilo CSS directamente al cuerpo (body) de la página
    st.markdown(page_bg_img, unsafe_allow_html=True)

background_image_path = "hosp.jpg"
background_opacity = 0.2
set_background(background_image_path)

## Crear la barra lateral con los botones
# with st.sidebar:
#     selection=option_menu(
#         menu_title="",
#         options = ['Portada', 'Dashboard', 'Predictor'],
#         icons = ["house-heart-fill", "calendar2-heart-fill","wrench"],
#         #menu_icon = "hospital-fill",
#         default_index=0
#     )


#Reducir el espacio en blanco al inicio de la página
st.markdown("""
        <style>
               .block-container {
                    padding-top: 6rem;
                    padding-bottom: 0rem;
                    padding-left: 3rem;
                    padding-right: 3rem;
                }
        </style>
        """, unsafe_allow_html=True)

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
    'change': 'Cambio en la Medicación de diabetes',
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
    st.subheader("Bienvenido a ⚕️DiabeRisk.")
    
# Slideshow de imagenes
    components.html(
        """
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    * {box-sizing: border-box;}
    body {font-family: Verdana, sans-serif;}
    .mySlides {display: none;}

    img {
        vertical-align: middle;
        object-fit: cover;
        height: 360px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.4); /* Sombra para el efecto de viñeta */
    }
    /* Slideshow container */
    .slideshow-container {
    max-width: 650px;
    position: relative;
    padding: 10px 10px 10px 10px;
    margin: auto;
    overflow: hidden;
    }

    /* Caption text */
    .text {
    color: #f2f2f2;
    font-size: 15px;
    padding: 0px 12px;
    position: absolute;
    bottom: 8px;
    width: 100%;
    text-align: center;
    }

    /* Number text (1/3 etc) */
    .numbertext {
    color: #f2f2f2;
    font-size: 12px;
    padding: 25px 12px;
    position: absolute;
    top: 0;
    }

    /* The dots/bullets/indicators */
    .dot {
    height: 10px;
    width: 10px;
    padding: -50px 2px;
    margin: 0px 2px;
    background-color: #bbb;
    border-radius: 50%;
    display: inline-block;
    transition: background-color 0.6s ease;
    }

    .active {
    background-color: #717171;
    }

    /* Fading animation */
    .fade {
    animation-name: fade;
    animation-duration: 4s;
    }

    @keyframes fade {
    0% { opacity: 0; } /* Comienza invisible */
    15% { opacity: 1; }
    50% { opacity: 1; } /* Opacidad máxima a la mitad */
    85% { opacity: 1; }
    100% { opacity: 0; } /* Desaparece al final */
    }

    /* On smaller screens, decrease text size */
    @media only screen and (max-width: 300px) {
    .text {font-size: 11px}
    }
    </style>
    </head>
    <body>

    <div class="slideshow-container">

    <div class="mySlides fade">
    <div class="numbertext">1 / 4</div>
    <img src="https://unsplash.com/photos/L8tWZT4CcVQ/download?ixid=M3wxMjA3fDB8MXxzZWFyY2h8MTh8fG1lZGljaW5lfGVzfDB8fHx8MTcxMzg4NDg1MHwy&force=true&w=2400" style="width:100%">
    <div class="text"></div>
    </div>

    <div class="mySlides fade">
    <div class="numbertext">2 / 4</div>
    <img src="https://unsplash.com/photos/7jjnJ-QA9fY/download?ixid=M3wxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNzE0MDAwMjY3fA&force=true&w=2400" style="width:100%">
    <div class="text"></div>
    </div>

    <div class="mySlides fade">
    <div class="numbertext">3 / 4</div>
    <img src="https://unsplash.com/photos/NFvdKIhxYlU/download?ixid=M3wxMjA3fDB8MXxzZWFyY2h8MTh8fGRvY3RvcnN8ZXN8MHx8fHwxNzE0MDAyOTYyfDI&force=true&w=2400" style="width:100%">
    <div class="text"></div>
    </div>

     <div class="mySlides fade">
    <div class="numbertext">4 / 4</div>
    <img src="    https://unsplash.com/photos/zQEmEAb-WpY/download?ixid=M3wxMjA3fDB8MXxzZWFyY2h8MjB8fGhvc3BpdGFsfGVzfDB8fHx8MTcxMzkzNDY5Mnww&force=true&w=2400" style="width:100%">
    <div class="text"></div>
    </div>

    </div>
    <br>

    <div style="text-align:center">
    <span class="dot"></span> 
    <span class="dot"></span> 
    <span class="dot"></span>
    <span class="dot"></span> 
    </div>

    <script>
    let slideIndex = 0;
    showSlides();

    function showSlides() {
    let i;
    let slides = document.getElementsByClassName("mySlides");
    let dots = document.getElementsByClassName("dot");
    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";  
    }
    slideIndex++;
    if (slideIndex > slides.length) {slideIndex = 1}    
    for (i = 0; i < dots.length; i++) {
        dots[i].className = dots[i].className.replace(" active", "");
    }
    slides[slideIndex-1].style.display = "block";  
    dots[slideIndex-1].className += " active";
    setTimeout(showSlides, 4000); // Change image every 4 seconds
    }
    </script>

    </body>
    </html> 

        """,
        height=460,
    )

    st.write('Elija la opción Dashboard para ver datos y visualizaciones, o Predictor para acceder a la herramienta predictora del reingreso de pacientes diabéticos.')

    # Crear tres columnas
    col1, col2, col3 = st.columns(3)
    
    # Columna 1
    with col1:
        st.subheader("👩🏻‍⚕️ :bar_chart: Analiza")
        st.write("Analiza los datos de los pacientes según su historial médico.")

    # Columna 2
    with col2:
        st.subheader(":computer: :chart_with_upwards_trend: Predice")
        st.write("Predice la probabilidad de readmisión de los pacientes.")

    # Columna 3
    with col3:
        st.subheader(":hospital: :bulb: Optimiza")
        st.write("Optimiza los recursos hospitalarios según las predicciones realizadas.")


    with st.expander("**Acerca de los datos**"):
        st.write("""
        El conjunto de datos representa diez años (1999-2008) de atención clínica en 130 hospitales y redes de prestación integradas de EE. UU. 
                 
        Cada fila corresponde a los registros hospitalarios de pacientes diagnosticados con diabetes, que se sometieron a análisis de laboratorio, medicamentos y permanecieron hasta 14 días. 
                 
        El objetivo es determinar el reingreso temprano del paciente. 
                 
        El problema es importante por las siguientes razones: 
                 
        A pesar de la evidencia de alta calidad que muestra mejores resultados clínicos para los pacientes diabéticos que reciben diversas intervenciones preventivas y terapéuticas, muchos pacientes no las reciben. Esto puede atribuirse en parte al manejo arbitrario de la diabetes en ambientes hospitalarios, que no atienden el control glucémico. 
                 
        No brindar una atención adecuada a la diabetes no solo aumenta los costos de gestión de los hospitales (a medida que los pacientes son readmitidos), sino que también afecta la morbilidad y mortalidad de los pacientes, que pueden enfrentar complicaciones asociadas con la diabetes.
        
        """)

        # Crear un enlace a la página de los datos
        st.markdown("[Fuente](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)")

    with st.expander("**Contexto**"):
                      
        team = {
            "Data Science": {
                "name": "Adrian Della Valentina",
                "linkedin": "[![Linkedin](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/in/adrian-della-valentina/)",
                "github": "[![Github](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/AdrianDVnqn/)"
            },
            "Data Analyst": {
                "name": "Daniel Menendez Gomez",
                "linkedin": "[![Linkedin](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=LinkedIn&logoColor=white)](enlace/al/que/deseas/ir)",
                "github": "[![Github](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=GitHub&logoColor=white)](enlace/al/que/deseas/ir)"
            },
            "ETL Developer": {
                "name": "Juan Mendoza Lopez",
                "linkedin": "[![Linkedin](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=LinkedIn&logoColor=white)](enlace/al/que/deseas/ir)",
                "github": "[![Github](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=GitHub&logoColor=white)](enlace/al/que/deseas/ir)"
            },
            "Analista BI": {
                "name": "Diego Suárez",
                "linkedin": "[![Linkedin](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/in/diego-suarez-escobar/)",
                "github": "[![Github](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=GitHub&logoColor=white)](https://www.linkedin.com/in/diego-suarez-escobar/)"
            }
        }

        st.markdown("""
                 
        Este proyecto fue realizado en el marco de la simulación laboral organizada por No Country.
        
        El grupo C17-77-FT-DATA-BI está conformado por:
                    
        Nombre         | Rol | LinkedIn | GitHub
        ------------- | ------------- | ------------- | -------------
        Adrian Della Valentina | Data Science | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/in/adrian-della-valentina/) | [![GitHub](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/AdrianDVnqn/)
        Guillermo Gallo García | Data Science | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/in/guillermo-patricio-gallo-garcia-0a3bb3bb/) | [![GitHub](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/Galo0000/)
        Daniel Menendez Gomez | Data Analyst | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/in/danielgomz/) | [![GitHub](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/danielGomz/)
        Juan Mendoza Lopez | ETL Developer | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/in/juan-mendoza00/) | [![GitHub](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/Juan-Mendoza00/)
        Diego Suárez | Analista BI | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/in/diego-suarez-escobar/) | [![GitHub](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/Dsuarezz20/)
        """)



########################
######################## DASHBOARD
########################



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



#########################
    #################### PREDICTOR
########################


else:

        # Cargar el DataFrame
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


    popup_results = Modal(key="results", title="Resultados")
    popup_save = Modal(key="save", title="Datos Guardados",max_width=3000)
    popup_excel = Modal(key="excel", title="Excel")

    columnas_resultados = list(variable_names.keys()) + ['readmitted', 'Probabilidad']
    resultados_df = pd.DataFrame(columns=columnas_resultados)
    lista_probabilidades = [0.1, 0.2, 0.3]
    # Inicializar resultados_df en la primera ejecución
    if 'resultados_df' not in st.session_state:
        st.session_state['resultados_df'] = pd.DataFrame(columns=columnas_resultados)


        #Funcion que borra datos
    def func_delete():
        for widget_key in categorical_cols:
            if widget_key in st.session_state:
                values = df[widget_key].unique()
                st.session_state[widget_key] = values[0] if len(values) > 0 else None
        for widget_key in numerical_cols:
            if widget_key in st.session_state:
                st.session_state[widget_key] = 0


    def realizar_prediccion(datos_prediccion, model):
      try:
        # Verificar si todos los datos necesarios están presentes
        #if any(pd.isnull(datos_prediccion[numerical_cols])):
        #  raise ValueError("Es necesario cargar todos los datos para realizar la predicción")
        # Realizar la predicción
        prediccion = model.predict(datos_prediccion)[0]
        probabilidad = model.predict_proba(datos_prediccion)[0][1]  # Probabilidad de la clase positiva (readmisión)
        # Calcular los valores SHAP
        # Obtener el modelo XGBoost del pipeline
        modelo_xgb = model.named_steps['classifier']
        # Preprocesamiento de datos
        # Escalar características numéricas y convertir características categóricas usando OneHotEncoder
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)  # drop='first' para evitar la multicolinealidad
        ])

        # Aplicar el preprocesamiento a X
        datos_preproc = preprocessor.fit_transform(datos_prediccion)
        
        feat_names = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

        # Convertir la matriz dispersa a un DataFrame de pandas
        datos_preproc_df = pd.DataFrame(datos_preproc)

        # Asignar los nuevos nombres de columnas a datos_prediccion
        datos_preproc_df.columns = feat_names

        # shap.initjs()
        # explainer = shap.Explainer(modelo_xgb)
        # shap_values = explainer.shap_values(datos_preproc)
        # shap.force_plot(explainer.expected_value[0], shap_values[0])

          # Mostrar el resultado de la predicción y la probabilidad
        if prediccion == 1:
            datos_prediccion['readmitted'] = prediccion
            datos_prediccion['Probabilidad'] = probabilidad
            with popup_results.container():
            #with st.sidebar:
                lista_probabilidades.append(probabilidad)
                st.markdown(f"🩺**Predicción: 🚨Hay readmisión del paciente.** Probabilidad: **{probabilidad*100:.1f}%**")
                st.dataframe(datos_prediccion)
                resultados_df = st.session_state.get('resultados_df', pd.DataFrame())
                # Concatenar el nuevo resultado al DataFrame existente
                resultados_df = pd.concat([resultados_df, datos_prediccion], ignore_index=True)
                # Guardar el DataFrame actualizado en st.session_state
                st.session_state['resultados_df'] = resultados_df
                # Mostrar mensaje de éxito
                st.write("Datos guardados correctamente en resultados_df")
                    #st.pyplot(shap.plots.force(shap_values[0]))
                    #st.dataframe(datos_preproc_df)
                               
        else:
            datos_prediccion['readmitted'] = prediccion
            datos_prediccion['Probabilidad'] = probabilidad
            with popup_results.container():
            #with st.sidebar:
                st.markdown(f"🩺**Predicción:** 🟢No hay readmisión del paciente. Probabilidad: **{probabilidad*100:.1f}%**")
                # Agregar los resultados a resultados_df
                st.dataframe(datos_prediccion)
                resultados_df = st.session_state.get('resultados_df', pd.DataFrame())
                # Concatenar el nuevo resultado al DataFrame existente
                resultados_df = pd.concat([resultados_df, datos_prediccion], ignore_index=True)
                # Guardar el DataFrame actualizado en st.session_state
                st.session_state['resultados_df'] = resultados_df
                # Mostrar mensaje de éxito
                st.write("Datos guardados correctamente en resultados_df")
                time.sleep(1)
            #st.pyplot(shap.plots.force(shap_values[0]))
            #st.dataframe(datos_preproc_df)
      except Exception as e:
            #with popup_results.container():
            with st.sidebar:
                st.markdown(f'⚠️Error al realizar la predicción: {e}')

    # Mostrar el DataFrame de datos del paciente y el botón de predicción

    #st.write("Datos del paciente:")
    #st.dataframe(datos_prediccion)
    
    options = {
        "show_sidebar": True,
    }

        # Inject custom CSS to set the width of the sidebar (200 px is the minimum possible)
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 200px !important; # Set the width to your desired value
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

        # Crear la barra lateral con los botones
    with st.sidebar:

        #Botón para borrar datos
        if st.button("**Borrar Datos**"):
            func_delete()

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
                    value = st.number_input(variable_names[col_name], step=1, min_value=0, help=descriptions.get(col_name, "Sin descripción"), key=col_name)
                    datos_prediccion[col_name] = [value]
                elif col_name in categorical_cols:
                    values = df[col_name].unique()
                    value = st.selectbox(variable_names[col_name], values, help=descriptions.get(col_name, "Sin descripción"), key=col_name)
                    datos_prediccion[col_name] = [value]




    # Color de fondo para la barra lateral (en este caso, rojo)
    sidebar_color = "#FF0000"

    # Aplicar el estilo CSS a la barra lateral
    st.markdown(
        f"""
        <style>
        .sidebar .sidebar-content {{
            background-color: {sidebar_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


    with st.sidebar:



        s = """
        <style>
        div.stButton > button:first-child { 
            border: 3px solid #42bda1; 
            border-radius: 15px 15px 15px 15px; 
            width: 100%;
            background-color: #a0cdde;
        }
        
        .stButton {
        display: flex;
        justify-content: flex-end;
        }
        </style>
        """
        st.markdown(s, unsafe_allow_html=True)

        # Botón para realizar la predicción
        if st.button("**Realizar Predicción**"):
                #popup.open()
                realizar_prediccion(datos_prediccion, model)
        
        st.markdown("#####")

        # Botón para ver dataframe de resultados guardados
        modal_guardados = st.button("**Datos Guardados**")
        if modal_guardados:
                popup_save.open()
        if popup_save.is_open():
                with popup_save.container():
                    resultados_df = st.session_state['resultados_df']
                    st.dataframe(resultados_df)

                    @st.experimental_memo
                    def convert_df_csv(df):
                       return df.to_csv(index=False).encode('utf-8')
                    
                    csv = convert_df_csv(resultados_df)
                    
                    st.download_button(
                       "Presiona para Descargar Archivo CSV 📄",
                       csv,
                       "diaberisk_resultados.csv",
                       "text/csv",
                       key='download-csv'
                    )

                    buffer = io.BytesIO()
                    def convert_df_xlsx(df):
                       return df.to_excel(index=False).encode('utf-8')
                    
                    # Create a Pandas Excel writer using XlsxWriter as the engine.
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        # Write each dataframe to a different worksheet.
                        resultados_df.to_excel(writer, sheet_name='Sheet1')
                        # Close the Pandas Excel writer and output the Excel file to the buffer
                        writer.save()

                        st.download_button(
                            label="Presiona para Descargar Archivo Excel 📊",
                            data=buffer,
                            file_name="diaberisk_resultados.xlsx",
                            mime="application/vnd.ms-excel"
                        )

                    # Agregar un botón para guardar en Excel
                   # if st.button("Guardar en archivo Excel"):
                    #        resultados_df = st.session_state['resultados_df']
                            # Guardar DataFrame en un archivo Excel
                    #        resultados_df.to_excel('resultados.xlsx', index=False)
                   #         # Mostrar mensaje de éxito
                   #         st.write("Datos guardados correctamente en resultados.xlsx")
                    #Para CSV
                   # if st.button("Guardar en archivo .csv"):
                     #       resultados_df = st.session_state['resultados_df']
                            # Guardar DataFrame en un archivo CSV
                      #      resultados_df.to_csv('resultados.csv', index=False)
                            # Mostrar mensaje de éxito
                      #      st.write("Datos guardados correctamente en resultados.csv")




        
        


