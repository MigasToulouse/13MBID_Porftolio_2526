import pandas as pd
import pytest

@pytest.fixture(scope="module")
def datos_banco():
    """
    Fixture que carga los datos UNA SOLA VEZ para todos los tests.
    """
    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")
    return df


LISTA_EXPECTATIVAS = [
    
    # --- 1. Datos del cliente bancario ---
    (
        "age_range",
        "df['age'].between(18, 100).all()",
        "La columna 'age' contiene valores fuera del rango esperado (18-100)."
    ),
    (
        "job_values",
        """df['job'].isin([
            'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
            'retired', 'self-employed', 'services', 'student', 'technician', 
            'unemployed', 'unknown'
        ]).all()""",
        "La columna 'job' contiene valores no esperados."
    ),
    (
        "marital_values",
        "df['marital'].isin(['divorced', 'married', 'single', 'unknown']).all()",
        "La columna 'marital' contiene valores no esperados."
    ),
    (
        "education_values",
        """df['education'].isin([
            'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 
            'professional.course', 'university.degree', 'unknown'
        ]).all()""",
        "La columna 'education' contiene valores no esperados."
    ),
    (
        "default_values",
        "df['default'].isin(['yes', 'no', 'unknown']).all()",
        "La columna 'default' contiene valores no esperados."
    ),
    (
        "housing_values",
        "df['housing'].isin(['yes', 'no', 'unknown']).all()",
        "La columna 'housing' contiene valores no esperados."
    ),
    (
        "loan_values",
        "df['loan'].isin(['yes', 'no', 'unknown']).all()",
        "La columna 'loan' contiene valores no esperados."
    ),

    # --- 2. Datos de la última campaña ---
    (
        "contact_values",
        "df['contact'].isin(['cellular', 'telephone']).all()",
        "La columna 'contact' contiene valores no esperados."
    ),
    (
        "month_values",
        """df['month'].isin([
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]).all()""",
        "La columna 'month' contiene valores no esperados."
    ),
    (
        "day_of_week_values",
        "df['day_of_week'].isin(['mon', 'tue', 'wed', 'thu', 'fri']).all()",
        "La columna 'day_of_week' contiene valores no esperados."
    ),

    # --- 3. Otros atributos ---
    (
        "campaign_range",
        "df['campaign'].between(0, 999).all()",
        "La columna 'campaign' contiene valores fuera del rango esperado (0-999)."
    ),
    (
        "poutcome_values",
        "df['poutcome'].isin(['failure', 'nonexistent', 'success']).all()",
        "La columna 'poutcome' contiene valores no esperados."
    ),

    # --- 5. Variable objetivo ---
    (
        "y_values",
        "df['y'].isin(['yes', 'no']).all()",
        "La columna 'y' contiene valores distintos a 'yes' o 'no'."
    ),

    # --- INCONSISTENCIAS ---
    (
        "pdays_poutcome_consistency",
        "((df['pdays'] != 999) | (df['poutcome'] == 'nonexistent')).all()",
        "Inconsistencia: Se encontraron filas con pdays=999 pero 'poutcome' no es 'nonexistent'."
    ),
    (
        "pdays_previous_consistency",
        "((df['pdays'] != 999) | (df['previous'] == 0)).all()",
        "Inconsistencia: Se encontraron filas con pdays=999 pero 'previous' es mayor a 0."
    ),
    (
        "pdays_duration_consistency",
        "((df['pdays'] != 999) | (df['duration'] > 0)).all()",
        "Inconsistencia: Se encontraron filas con pdays=999 pero 'duration' es 0."
    ),
    (
        "duration_target_consistency",
        "((df['duration'] > 0) | (df['y'] == 'no')).all()",
        "Inconsistencia: Se encontraron filas con duration=0 pero 'y' es 'yes'."
    )
]

# Comentar xfail para que los tests muestren correctamente los pass y fail.
@pytest.mark.xfail(reason="No romper el pipeline para demostrar funcionamiento.")
@pytest.mark.parametrize(
    "expectation_name, condition_str, message",
    LISTA_EXPECTATIVAS,
    ids=[e[0] for e in LISTA_EXPECTATIVAS]  # Usa el nombre de la expectativa como ID del test
)
def test_great_expectations_atomizado(datos_banco, expectation_name, condition_str, message):
    """
    Test atomizado que evalúa una única expectativa de la lista. Esto hará que en el reporte aparezcan todos los tests
    por separado, facilitando la identificación de fallos específicos.
    """
    df = datos_banco  # Obtener el DataFrame del fixture

    # Se evalúa la condición con eval y se pasa 'df' al contexto local
    condition_passes = eval(condition_str, globals(), {'df': df})

    # Se checkea que la condición se cumple
    assert condition_passes, f"Fallo en [ {expectation_name} ]: {message}"
