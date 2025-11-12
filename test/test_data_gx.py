import pandas as pd

def test_great_expectations():
    """Test para verificar que los datos cumplen con las expectativas definidas en Great Expectations.
    
    Raises:
        AssertionError: Si alguna de las expectativas no se cumple.
    """

    # Cargar el DataFrame desde un archivo CSV
    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")

    results = {
        "success": True,
        "expectations": [],
        "statistics": {"success_count": 0, "total_count": 0}
    }

    def add_expectation(expectation_name, condition, message=""):
        
        results["statistics"]["total_count"] += 1
        if condition:
            results["statistics"]["success_count"] += 1
            results["expectations"].append({
                "expectation": expectation_name,
                "success": True,
            })
        else:
            results["success"] = False
            results["expectations"].append({
                "expectation": expectation_name,
                "success": False,
                "message": message
            })
    
    # Validaciones de valores sobre atributos

    # --- 1. Datos del cliente bancario ---
    add_expectation(
        "age_range",
        df["age"].between(18, 100).all(), # Suponemos que se espera mayoría de edad.
        "La columna 'age' contiene valores fuera del rango esperado (18-100)."
    )

    add_expectation(
    "job_values",
    df["job"].isin([
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
        "retired", "self-employed", "services", "student", "technician", 
        "unemployed", "unknown"
    ]).all(),
    "La columna 'job' contiene valores no esperados."
    )

    add_expectation(
        "marital_values",
        df["marital"].isin(["divorced", "married", "single", "unknown"]).all(),
        "La columna 'marital' contiene valores no esperados."
    )

    add_expectation(
        "education_values",
        df["education"].isin([
            "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", 
            "professional.course", "university.degree", "unknown"
        ]).all(),
        "La columna 'education' contiene valores no esperados."
    )

    add_expectation(
        "default_values",
        df["default"].isin(["yes", "no", "unknown"]).all(),
        "La columna 'default' contiene valores no esperados."
    )

    add_expectation(
        "housing_values",
        df["housing"].isin(["yes", "no", "unknown"]).all(),
        "La columna 'housing' contiene valores no esperados."
    )

    add_expectation(
        "loan_values",
        df["loan"].isin(["yes", "no", "unknown"]).all(),
        "La columna 'loan' contiene valores no esperados."
    )

    # --- 2. Datos de la última campaña ---

    add_expectation(
        "contact_values",
        df["contact"].isin(["cellular", "telephone"]).all(),
        "La columna 'contact' contiene valores no esperados."
    )

    add_expectation(
        "month_values",
        df["month"].isin([
            "jan", "feb", "mar", "apr", "may", "jun", 
            "jul", "aug", "sep", "oct", "nov", "dec"
        ]).all(),
        "La columna 'month' contiene valores no esperados."
    )

    add_expectation(
        "day_of_week_values",
        df["day_of_week"].isin(["mon", "tue", "wed", "thu", "fri"]).all(),
        "La columna 'day_of_week' contiene valores no esperados."
    )

    # --- 3. Otros atributos ---

    add_expectation(
        "campaign_range",
        (df["campaign"] >= 1).all(), # Al menos 1 contacto en esta campaña
        "La columna 'campaign' contiene valores no positivos (debe ser >= 1)."
    )

    add_expectation(
        "campaign_range",
        df["campaign"].between(0, 999).all(), # Rango esperado de contactos
        "La columna 'campaign' contiene valores fuera del rango esperado (0-999)."
    )

    add_expectation(
        "previous_range",
        (df["previous"] >= 0).all(),
        "La columna 'previous' contiene valores negativos."
    )

    add_expectation(
        "poutcome_values",
        df["poutcome"].isin(["failure", "nonexistent", "success"]).all(),
        "La columna 'poutcome' contiene valores no esperados."
    )

    # --- 5. Variable objetivo ---

    add_expectation(
        "y_values",
        df["y"].isin(["yes", "no"]).all(),
        "La columna 'y' contiene valores distintos a 'yes' o 'no'."
    )
