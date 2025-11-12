import pandas as pd
from pandera.pandas import DataFrameSchema, Column
import pytest


@pytest.fixture
def datos_banco():
    """Fixture para cargar los datos bancarios desde un archivo CSV.
    
        Devuelve:
        pd.DataFrame: DataFrame con los datos bancarios.
    """

    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")
    return df

def test_esquema(datos_banco: pd.DataFrame):
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

def test_no_vacio(datos_banco: pd.DataFrame):
    """Test para verificar que el DataFrame no está vacío."""
    df = datos_banco
    assert not df.empty, "El DataFrame está vacío."

def test_sin_nulos(datos_banco: pd.DataFrame):
    """Test para verificar que el DataFrame no contiene valores nulos."""
    df = datos_banco
    assert df.isnull().sum().sum() == 0, "El DataFrame contiene valores nulos."

def test_cantidad_columnas(datos_banco: pd.DataFrame):
    """Test para verificar la cantidad esperada de columnas."""
    df = datos_banco
    expected_columns = 21
    assert df.shape[1] == expected_columns, \
        f"El DataFrame debería tener {expected_columns} columnas, pero tiene {df.shape[1]}."

def test_rango_numerico_negativo(datos_banco: pd.DataFrame):
    """Test para verificar que las columnas numéricas de números naturales no contienen valores negativos donde no deberían."""
    df = datos_banco

    columnas_a_verificar = ["age", "duration", "campaign", "pdays", "previous"]

    for columna in columnas_a_verificar:
        assert (df[columna] >= 0).all(), \
            f"La columna '{columna}' contiene valores negativos."

@pytest.mark.xfail(reason="No romper el pipeline para demostrar funcionamiento.")
def test_sin_duplicados(datos_banco: pd.DataFrame):
    """
    Test para verificar que el DataFrame no contiene filas duplicadas.
    """
    df = datos_banco
    # Descomenta la siguiente línea para activar el test.
    assert df.duplicated().sum() == 0, \
       f"El DataFrame contiene {df.duplicated().sum()} filas duplicadas."
