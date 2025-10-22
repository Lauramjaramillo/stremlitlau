"""
===============================================================================
Proyecto: Clasificación de Pingüinos 🐧
Autor: Laura Jaramillo
Descripción:
    Este script entrena un modelo de aprendizaje automático utilizando el 
    conjunto de datos de pingüinos. El objetivo es predecir la especie de 
    un pingüino a partir de sus características físicas y su ubicación.

    El modelo entrenado se guarda en un archivo .pickle para ser reutilizado 
    en una aplicación Streamlit interactiva.
===============================================================================
"""

# =============================================================================
# 📦 Importación de librerías
# =============================================================================
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle  # Para guardar y cargar modelos entrenados

# =============================================================================
# 📂 Carga y preparación de datos
# =============================================================================
def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV y elimina filas con valores nulos.

    Parámetros
    ----------
    ruta_csv : str
        Ruta del archivo CSV que contiene los datos de pingüinos.

    Retorna
    -------
    df : pd.DataFrame
        DataFrame limpio y preparado.
    """
    df = pd.read_csv(ruta_csv)
    df.dropna(inplace=True)
    return df


# =============================================================================
# ⚙️ Preparación de variables
# =============================================================================
def preparar_datos(df: pd.DataFrame):
    """
    Separa las variables de entrada (X) y salida (y),
    y convierte las variables categóricas en numéricas.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales.

    Retorna
    -------
    X : pd.DataFrame
        Variables de entrada codificadas.
    y : pd.Series
        Variable de salida (especie) codificada.
    uniques : list
        Lista con las especies originales.
    """
    # Variable de salida
    y = df['species']

    # Variables predictoras
    X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
            'body_mass_g', 'island', 'sex']]

    # Codificación de variables categóricas (One-Hot Encoding)
    X = pd.get_dummies(X)

    # Codificación de las especies (factorización)
    y, uniques = pd.factorize(y)

    return X, y, uniques


# =============================================================================
# 🧠 Entrenamiento del modelo
# =============================================================================
def entrenar_modelo(X, y):
    """
    Entrena un modelo Random Forest con los datos de entrada.

    Parámetros
    ----------
    X : pd.DataFrame
        Variables de entrada.
    y : pd.Series
        Variable objetivo.

    Retorna
    -------
    model : RandomForestClassifier
        Modelo de bosque aleatorio entrenado.
    accuracy : float
        Precisión del modelo sobre el conjunto de prueba.
    """
    # División de los datos (80% prueba, 20% entrenamiento)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    # Creación y entrenamiento del modelo
    model = RandomForestClassifier(random_state=15)
    model.fit(x_train, y_train)

    # Evaluación
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)

    return model, accuracy


# =============================================================================
# 💾 Guardado del modelo y etiquetas
# =============================================================================
def guardar_modelo(model, uniques, ruta_modelo='random_forest_penguin.pickle',
                   ruta_salida='output_penguin.pickle'):
    """
    Guarda el modelo entrenado y las etiquetas codificadas en archivos .pickle.

    Parámetros
    ----------
    model : RandomForestClassifier
        Modelo entrenado a guardar.
    uniques : list
        Lista de nombres de especies originales.
    ruta_modelo : str
        Nombre o ruta donde se guardará el modelo.
    ruta_salida : str
        Nombre o ruta donde se guardarán las etiquetas.
    """
    # Guardar modelo
    with open(ruta_modelo, 'wb') as f:
        pickle.dump(model, f)

    # Guardar etiquetas de salida
    with open(ruta_salida, 'wb') as f:
        pickle.dump(uniques, f)


# =============================================================================
# 🚀 Ejecución principal
# =============================================================================
if __name__ == "__main__":
    print("🔹 Iniciando entrenamiento del modelo de pingüinos...\n")

    # 1️⃣ Cargar datos
    df = cargar_datos('penguins.csv')
    print("✅ Datos cargados correctamente. Filas:", df.shape[0])

    # 2️⃣ Preparar variables
    X, y, uniques = preparar_datos(df)
    print("✅ Variables preparadas correctamente.")
    print("   Especies detectadas:", list(uniques))

    # 3️⃣ Entrenar modelo
    modelo, precision = entrenar_modelo(X, y)
    print(f"✅ Modelo entrenado con precisión: {precision:.4f}")

    # 4️⃣ Guardar modelo y etiquetas
    guardar_modelo(modelo, uniques)
    print("✅ Archivos guardados:")
    print("   - random_forest_penguin.pickle")
    print("   - output_penguin.pickle")

    print("\n🎉 Entrenamiento completado exitosamente.")
