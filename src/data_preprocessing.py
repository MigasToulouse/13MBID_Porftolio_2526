import pandas as pd
import numpy as np

INPUT_CSV = 'data/raw/bank-additional-full.csv'
OUTPUT_CSV = 'data/processed/bank-additional-full_preprocessed.csv'
TRANSFORMATIONS_FILE = 'docs/transformaciones_dataset.md'

transformations_doc = ["# Transformaciones realizadas en el dataset\n\n"]

def add_to_doc(description: str = "") -> None:
    """
    Agrega una descripción de una transformación a la lista de transformaciones.

    Parámetros:
    - description: Descripción de la transformación realizada.
    """

    transformations_doc.append(f"{description}\n")

def write_transformations_doc(doc_path: str = TRANSFORMATIONS_FILE, shape: tuple = None) -> None:
    """
    Escribe la lista completa de transformaciones en el documento especificado,
    sobrescribiéndolo con la numeración correcta.
    """
    with open(doc_path, 'w', encoding='utf-8') as f:
        # Primero el título
        f.write(transformations_doc[0]) 
        
        # Escribir el resto como una lista numerada
        for i, description in enumerate(transformations_doc[1:], start=1):
            f.write(f"{i}. {description}")

        # Por último, agregar información del shape del DataFrame resultante
        f.write(f"\nEl DataFrame resultante tiene {shape[0]} filas y {shape[1]} columnas.\n")

def preprocess_data(input_path : str = INPUT_CSV, output_path: str = OUTPUT_CSV, ) -> tuple:
    """
    Función que preprocesa los datos del banco realizando eliminaciones y transformaciones necesarias.

    Parámetros:
    - input_path: Ruta al archivo CSV de entrada.
    - output_path: Ruta al archivo CSV de salida donde se guardarán los datos procesados
    """
    # Cargar los datos
    df = pd.read_csv(input_path, sep=';')

    # LIMPIEZA
    df.columns = df.columns.str.replace('.', '_')
    add_to_doc("Se adaptaron los nombres de las columnas al formato snake_case.")
    
    df.drop(columns=['default'], inplace=True)
    add_to_doc("Se eliminó la columna 'default' por tener demasiados valores desconocidos.")

    df.drop(columns=['pdays'], inplace=True)
    add_to_doc("Se eliminó la columna 'pdays' por poca variabilidad e incoherencia con 'duration'.")

    df.replace('unknown', np.nan, inplace=True)
    add_to_doc("Se reemplazaron los valores 'unknown' por NaN para su posterior eliminación.")

    df.dropna(inplace=True)
    add_to_doc("Se eliminaron las filas con valores NaN.")

    df.drop_duplicates(inplace=True)
    add_to_doc("Se eliminaron los registros duplicados.")

    # CONSTRUCCION
    # Diferencia entre contactos en campaña actual y anterior
    df['contacts_diff'] = df['campaign'] - df['previous']
    add_to_doc("Se creó una nueva columna 'contacts_diff' que representa la diferencia entre contactos en la campaña actual y la anterior.")

    # Se añade contact_before, un atributo que aparecerá en los nuevos datos en la fase modelado
    df['contacted_before'] = np.where(df['previous'] > 0, 1, 0)
    add_to_doc("Se creó una nueva columna 'contact_before' que indica si el cliente fue contactado en campañas anteriores.")

    # FORMATEO
    df['nr_employed'] = pd.to_numeric(df['nr_employed'], downcast='integer')
    add_to_doc("Se transformó la columna 'nr_employed' de float a entero por coherencia semántica.")

    binary_columns = ['housing', 'loan', 'y']
    for col in binary_columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    add_to_doc("Se binarizaron las columnas categóricas de dos clases ('housing', 'loan', 'y') a valores numéricos (1 para 'yes' y 0 para 'no').")

    # Guardar el DataFrame procesado
    df.to_csv(output_path, index=False, sep=';')

    return df.shape

if __name__ == "__main__":
    res = preprocess_data()
    write_transformations_doc(shape=res)