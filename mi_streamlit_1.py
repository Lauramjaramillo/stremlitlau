"""
===============================================================================
 PROYECTO: Tablero interactivo de análisis y predicción de pingüinos 🐧
 AUTORA: Laura Jaramillo
 CURSO: Análisis de Datos Avanzado - Talento Tech Innovador
 DESCRIPCIÓN:
     Esta aplicación web, creada con Streamlit, permite:
        1. Visualizar los datos del conjunto de pingüinos.
        2. Explorar gráficamente las relaciones entre variables.
        3. Utilizar un modelo de Machine Learning (Random Forest)
           para predecir la especie de un pingüino según sus características.
===============================================================================
"""

# ==============================
# 📦 IMPORTACIÓN DE LIBRERÍAS
# ==============================
import streamlit as st             # Interfaz web interactiva
import pandas as pd                # Análisis y manipulación de datos
import matplotlib.pyplot as plt    # Visualización básica
import seaborn as sns              # Visualización avanzada y estética
import sklearn                     # Librería para machine learning
import joblib                      # Para cargar modelos guardados (.pickle/.joblib)


# ==============================
# ⚙️ CONFIGURACIÓN DE LA PÁGINA
# ==============================
st.set_page_config(
    layout='centered',               # Centra el contenido
    page_title='Talento Tech Innovador',  # Título de la pestaña
    page_icon=':grinning:'           # Emoji en la pestaña
)

# Encabezado principal con dos columnas
t1, t2 = st.columns([0.3, 0.7])
# t1.image('./index.jpg', width=180)  # Imagen opcional
t2.title('Mi primer tablero')
t2.markdown('**tel:** 123 **| email:** talentotech@gmail.com')


# ==============================
# 🧭 CREACIÓN DE SECCIONES (TABS)
# ==============================
steps = st.tabs([
    'Introducción', 
    'Visualización de datos', 
    'Modelo ML', 
    '$\int_{-\infty}^\infty e^{\sigma\mu}dt$'  # pestaña decorativa
])


# ==============================
# 🟢 SECCIÓN 1: INTRODUCCIÓN
# ==============================
with steps[0]:
    st.title('Metadata')
    st.write('Bienvenido a mi proyecto de análisis de datos de pingüinos 🐧')

    # Cargar el dataset
    df = pd.read_csv('penguins.csv')

    # Mostrar las primeras filas de los datos
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())

# Mostrar las columnas disponibles (solo en la consola, útil para depuración)
print(df.columns)


# ==============================
# 🟡 SECCIÓN 2: VISUALIZACIÓN DE DATOS
# ==============================
with steps[1]:
    st.markdown('### Gráfica de los tipos de Pingüinos')

    # Selectores interactivos
    species = st.selectbox(
        'Seleccione la especie a visualizar',
        ['Adelie', 'Gentoo', 'Chinstrap']
    )

    x = st.selectbox('Seleccione la variable para el eje X', list(df.columns))
    y = st.selectbox('Seleccione la variable para el eje Y', list(df.columns))

    # Crear el gráfico
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x], y=df[y], data=df, hue='species', ax=ax)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f'Relación entre {x} y {y}')

    # Mostrar el gráfico en la app
    st.pyplot(fig)


# ==============================
# 🔵 SECCIÓN 3: MODELO DE MACHINE LEARNING
# ==============================
with steps[2]:
    st.markdown("### Predicción de la especie del pingüino 🧠")

    # Cargar modelo previamente entrenado y mapeo de etiquetas
    rfc = joblib.load('random_forest_penguin.pickle')
    unique_penguin_mapping = joblib.load('output_penguin.pickle')

    # Mostrar el modelo (opcional, solo para ver estructura)
    st.write("Modelo cargado correctamente:", rfc)

    # --- Entradas del usuario ---
    island = st.selectbox('Isla', options=['Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('Sexo', options=['Female', 'Male'])
    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body Mass (g)', min_value=0)

    st.write('Datos ingresados:', 
             [island, sex, bill_length, bill_depth, flipper_length, body_mass])

    # --- Codificación manual de variables categóricas ---
    island_biscoe, island_dream, island_torgerson = 0, 0, 0
    if island == 'Biscoe':
        island_biscoe = 1
    elif island == 'Dream':
        island_dream = 1
    elif island == 'Torgerson':
        island_torgerson = 1

    sex_female, sex_male = 0, 0
    if sex == 'Female':
        sex_female = 1
    elif sex == 'Male':
        sex_male = 1

    # --- Predicción con el modelo ---
    new_prediction = rfc.predict([[
        bill_length, bill_depth, flipper_length, body_mass,
        island_biscoe, island_dream, island_torgerson,
        sex_female, sex_male
    ]])

    # Obtener el nombre de la especie correspondiente
    prediction_species = unique_penguin_mapping[new_prediction][0]

    # Mostrar el resultado final
    st.success(f'✅ La especie del pingüino es **{prediction_species}**')


# ==============================
# 🧩 SECCIÓN 4: EXPRESIÓN MATEMÁTICA
# ==============================
with steps[3]:
    st.latex(r"\int_{-\infty}^\infty e^{\sigma\mu}dt")
    st.write("Esta pestaña es decorativa, solo muestra una expresión matemática.")

#Para correr python -m streamlit run "C:\Users\ZENBOOK\OneDrive - MUNICIPIO DE ENVIGADO\Documentos\Personales\Analisis de datos avanzado\Streamlit\mi_streamlit_1.py"
