"""
Script para entrenar un modelo de clasificación utilizando la técnica con mejor rendimiento
que fuera seleccionada durante la experimentación.
"""

# Utilidades
from os.path import join
from os import getcwd
import json
import argparse
from pathlib import Path
import joblib

# Gestión de dataframes
import pandas as pd

# Gestión de logs y modelos con MLFlow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Modelado y evaluación
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Visualización
import matplotlib.pyplot as plt

# PARAMETERS
MODEL_NAME = "Decision Tree Classifier"
MODEL_OUTPUT_PATH = join(getcwd(), "models", "decision_tree_model.pkl")
PREPROCESSOR_OUTPUT_PATH = join(getcwd(), "models", "preprocessor.pkl")
METRICS_OUTPUT_PATH = join(getcwd(), "metrics", "model_metrics.json")
DATA_PATH = join(getcwd(), "data", "processed", "bank-additional-full_preprocessed.csv")

def load_data(path: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV.
    
    Parámetros:
        path (str): Ruta al archivo CSV.
    Retorna:
        pd.DataFrame: Datos cargados.
    """

    df = pd.read_csv(path, sep=';')
    X = df.drop('y', axis=1)
    y = df['y']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def create_preprocessor(X_train: pd.DataFrame) -> list[ColumnTransformer, pd.DataFrame]:
    """
    Crea el preprocesador para los datos.
    
    Parámetros:
        X_train (pd.DataFrame): Datos de entrenamiento.
    Retorna:
        list[ColumnTransformer, pd.DataFrame]: Preprocesador y datos de entrenamiento modificados
    """

    # Evitar SettingWithCopyWarning    
    X_train = X_train.copy()

    # Convertir int a float para evitar problemas con RobustScaler
    int_columns = X_train.select_dtypes(include='int').columns
    X_train[int_columns] = X_train[int_columns].astype('float')

    # Identificación de columnas numéricas y categóricas
    numerical_columns=X_train.select_dtypes(exclude='object').columns
    categorical_columns=X_train.select_dtypes(include='object').columns

    # Pipeline para valores numéricos
    num_pipeline = Pipeline(steps=[
        ('RobustScaler', RobustScaler())
    ])

    # Pipeline para valores categóricos
    cat_pipeline = Pipeline(steps=[
        ('OneHotEncoder', OneHotEncoder(drop='first',sparse_output=False))
    ])

    # Se configuran los preprocesadores
    preprocessor_full = ColumnTransformer([
        ('num_pipeline', num_pipeline, numerical_columns),
        ('cat_pipeline', cat_pipeline, categorical_columns)
    ]).set_output(transform='pandas')
    
    return preprocessor_full, X_train

def balance_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Balancea el conjunto de datos mediante sobremuestreo de la clase minoritaria.
    
    Parámetros:
        X (pd.DataFrame): Características.
        y (pd.Series): Etiquetas.
    Retorna:
        tuple: Datos balanceados (X_balanced, y_balanced).
    """

    # Combinar los datos preprocesados con las etiquetas
    train_data = X.copy()
    train_data['target'] = y.reset_index(drop=True)

    # Separar por clase
    class_0 = train_data[train_data['target'] == 0]
    class_1 = train_data[train_data['target'] == 1]

    # Encontrar la clase minoritaria
    min_count = min(len(class_0), len(class_1))

    # Submuestreo balanceado - tomar una muestra igual al tamaño de la clase minoritaria
    class_0_balanced = resample(class_0, n_samples=min_count, random_state=42)
    class_1_balanced = resample(class_1, n_samples=min_count, random_state=42)

    # Combinar las clases balanceadas
    balanced_data = pd.concat([class_0_balanced, class_1_balanced])

    # Separar características y objetivo
    x_train_resampled = balanced_data.drop('target', axis=1)
    y_train_resampled = balanced_data['target']

    return x_train_resampled, y_train_resampled

def train_model(
        data_path: str = DATA_PATH,
        model_output_path: str = MODEL_OUTPUT_PATH,
        preprocessor_output_path: str = PREPROCESSOR_OUTPUT_PATH,
        metrics_output_path: str = METRICS_OUTPUT_PATH
    ) -> None:

    """
    Método principal para entrenar el modelo de clasificación.

    Parámetros:
        data_path (str): Ruta al archivo CSV con los datos preprocesados.
        model_output_path (str): Ruta para guardar el modelo entrenado.
        preprocessor_output_path (str): Ruta para guardar el preprocesador.
        metrics_output_path (str): Ruta para guardar las métricas del modelo.
    """
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Proyecto 13MBID-ABR2526 - Producción")

    with mlflow.start_run(run_name=f"{MODEL_NAME}"):

        print("Cargando datos...")
        X_train, X_test, y_train, y_test = load_data(data_path)
        print("Creando preprocesador...")
        preprocessor_full, X_train = create_preprocessor(X_train)

        # Convertir columnas enteras en X_test también
        int_columns = X_test.select_dtypes(include=['int', 'int64', 'int32']).columns
        X_test[int_columns] = X_test[int_columns].astype('float')

        print("Preprocesando datos de entrenamiento...")
        X_train_prep = preprocessor_full.fit_transform(X_train)
        X_test_prep = preprocessor_full.transform(X_test)

        print("Balanceando datos de entrenamiento...")
        X_train_balanced, y_train_balanced = balance_data(X_train_prep, y_train)

        print(f"Tamaño original: {len(X_train_prep)}")
        print(f"Tamaño balanceado: {len(X_train_balanced)}")
        print(f"Distribución: {y_train_balanced.value_counts().to_dict()}")

        print("Entrenando el modelo...")
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_balanced, y_train_balanced)

        print("Evaluando el modelo...")
        y_pred = model.predict(X_test_prep)

        # Crear pipeline completo
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor_full),
            ('model', model)
        ])

        # Crear signatures y ejemplos de entrada
        raw_input_example = X_test.iloc[:5]
        preprocessed_input_example = X_train_prep.iloc[:5]

        # Signature para el pipeline completo
        pipeline_signature = infer_signature(
            X_train, # Entrada sin preprocesar
            y_pred   # Predicciones del modelo
        )

        # Signature para el preprocesador
        preprocessor_signature = infer_signature(
            X_train,        # Entrada sin preprocesar
            X_train_prep    # Datos preprocesados
        )

        # Signature para el modelo
        model_signature = infer_signature(
            X_train_prep,   # Datos preprocesados
            y_pred          # Predicciones del modelo
        )

        # Cálculo de métricas
        metrics = {
            "f1_score": f1_score(y_test, y_pred),
            "recall_score": recall_score(y_test, y_pred),
            "precision_score": precision_score(y_test, y_pred),
            "accuracy_score": accuracy_score(y_test, y_pred)
        }

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        # Registro de parámetros
        mlflow.log_params({
            "model_type": MODEL_NAME,
            "n_neighbors": 7,
            "balancing_method": "undersampling",
            "train_samples": len(X_train_balanced),
            "test_samples": len(X_test)
        })

        # Registrar métricas
        mlflow.log_metrics(metrics)

        # Registrar matriz de confusión
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(ax=ax)
        plt.title("Matriz de Confusión - Modelo de producción")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()

        # Registrar el pipeline completo
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            signature=pipeline_signature,
            # input_example=raw_input_example
        )

        # Registrar el preprocesador
        mlflow.sklearn.log_model(
            sk_model=preprocessor_full,
            artifact_path="preprocessor",
            signature=preprocessor_signature,
            # input_example=raw_input_example
        )

        # Registrar el modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="Decision_Tree_Model",
            signature=model_signature,
            # input_example=preprocessed_input_example
        )

        # Guardar modelos localmente
        print("\nGuardando modelos...")
        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(preprocessor_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_output_path)
        print("Modelo guardado en:", model_output_path)
        joblib.dump(preprocessor_full, preprocessor_output_path)
        print("Preprocesador guardado en:", preprocessor_output_path)

        with open(metrics_output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print("Métricas guardadas en:", metrics_output_path)

        return model, preprocessor_full, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo KNN de clasificación.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DATA_PATH,
        help="Ruta al archivo CSV con los datos preprocesados."
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=MODEL_OUTPUT_PATH,
        help="Ruta para guardar el modelo entrenado."
    )
    parser.add_argument(
        "--preprocessor-output",
        type=str,
        default=PREPROCESSOR_OUTPUT_PATH,
        help="Ruta para guardar el preprocesador."
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=METRICS_OUTPUT_PATH,
        help="Ruta para guardar las métricas del modelo."
    )

    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        model_output_path=args.model_output,
        preprocessor_output_path=args.preprocessor_output,
        metrics_output_path=args.metrics_output
    )
