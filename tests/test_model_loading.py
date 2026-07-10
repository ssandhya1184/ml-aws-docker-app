from pathlib import Path
import joblib

def test_model_loads_successfully():
    """
    Verify that the trained model exists 
    and loads successfully

    """

    model_path = Path("artifacts/attrition_pipeline.pkl")

    # Arrange
    assert model_path.exists(), "Model file not found"

    # Act
    model = joblib.load(model_path)

    # Assert
    assert model is not None