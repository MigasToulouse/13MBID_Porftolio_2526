import pandas as pd
from pandera.pandas import DataFrameSchema, Column
import pytest
from pathlib import Path
from os.path import join
from os import getcwd
from datetime import datetime


@pytest.fixture
def datos_banco():
    """Fixture para cargar los datos bancarios desde un archivo CSV.
    
        Devuelve:
        pd.DataFrame: DataFrame con los datos bancarios.
    """

    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")
    return df

def test_esquema(datos_banco):
    """Test para validar el esquema del DataFrame de datos bancarios."""

    df = datos_banco

    esquema = DataFrameSchema({
        "age": Column(int, nullable=False),
        "job": Column(str, nullable=False),
        "marital": Column(str, nullable=False),
        "education": Column(str, nullable=False),
        "default": Column(str, nullable=False),
        "housing": Column(str, nullable=False),
        "loan": Column(str, nullable=False),
        "contact": Column(str, nullable=False),
        "month": Column(str, nullable=False),
        "day_of_week": Column(str, nullable=False),
        "duration": Column(int, nullable=False),
        "campaign": Column(int, nullable=False),
        "pdays": Column(int, nullable=False),
        "previous": Column(int, nullable=False),
        "poutcome": Column(str, nullable=False),
        "emp.var.rate": Column(float, nullable=False),
        "cons.price.idx": Column(float, nullable=False),
        "cons.conf.idx": Column(float, nullable=False),
        "euribor3m": Column(float, nullable=False),
        "nr.employed": Column(float, nullable=False),
        "y": Column(str, nullable=False)
    })

    esquema.validate(df)

def test_basico(datos_banco):
    """Test básico para verificar que el DataFrame no está vacío y tiene las columnas esperadas."""

    df = datos_banco

    # Verificar que el DataFrame no esté vacío
    assert not df.empty, "El DataFrame está vacío."
    # Verificar nulos
    assert df.isnull().sum().sum() == 0, "El DataFrame contiene valores nulos."
    # Verificar duplicados
    assert df.duplicated().sum() == 0, "El DataFrame contiene filas duplicadas."
    # Verificar cantidad de columnas
    expected_columns = 21
    assert df.shape[1] == expected_columns, f"El DataFrame debería tener {expected_columns} columnas, pero tiene {df.shape[1]}."

if __name__ == "__main__":

    TEST_DIR = join(getcwd(), "docs", "test_results")
    Path(TEST_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        test_esquema(datos_banco())
        test_basico(datos_banco())
        print("Todos los tests pasaron exitosamente.")
        with open(join(TEST_DIR, "test_success.log"), "w", encoding="utf-8") as log_file:
            log_file.write(f"{datetime.now()} Todos los tests pasaron exitosamente.\n")
    except AssertionError as e:
        print(f"Fallo en los tests: {e}")
        with open(join(TEST_DIR, "test_failures.log"), "w", encoding="utf-8") as log_file:
            log_file.write(f"{datetime.now()} Test fallido: {e}\n")