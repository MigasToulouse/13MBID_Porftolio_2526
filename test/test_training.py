import json
from pathlib import Path
import sys
import pytest


def test_training_metrics_regression(tmp_path):

    # 1. Calcular la ruta raíz
    project_root = Path(__file__).parent.resolve().parents[0]

    baseline_path = project_root / "metrics" / "model_metrics.json"
    if not baseline_path.exists():
        pytest.skip(f"No se encontró la baseline de métricas de entrenamiento. baseline_path: {baseline_path}")

    # Cargar métricas baseline
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    # 2. Añadir 'src' al path
    sys.path.insert(0, str(project_root / "src"))
    # Ahora puedo importar train_model
    from train_model import train_model 

    # Ejecutar entrenamiento
    data_path = project_root / "data" / "processed" / "bank-additional-full_preprocessed.csv"
    model_output_path = tmp_path / "knn_model.pkl"
    preprocessor_output_path = tmp_path / "preprocessor.pkl"
    metrics_output_path = tmp_path / "model_metrics.json"

    _,_,metrics = train_model(
        data_path=str(data_path),
        model_output_path=str(model_output_path),
        preprocessor_output_path=str(preprocessor_output_path),
        metrics_output_path=str(metrics_output_path)
    )

    # Comparar métricas
    assert set(metrics.keys()) == set(baseline.keys()), "Las claves de las métricas no coinciden con la baseline."
    atol = 1e-9
    for k in baseline.keys():
        assert metrics[k] == pytest.approx(baseline[k], rel = 0 ,abs=atol), (
        f"Métrica '{k}' difiere de la baseline: {metrics[k]} vs {baseline[k]}")
