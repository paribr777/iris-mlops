import joblib
import pandas as pd
import pytest
import os

VERSIONS = ["v0", "v1", "v2"]

@pytest.mark.parametrize("version", VERSIONS)
def test_model_load(version):
    model_path = f"models/iris_best_{version}.joblib"
    assert os.path.exists(model_path), f"{model_path} not found"
    
    model = joblib.load(model_path)
    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]], 
        columns=["sepal_length","sepal_width","petal_length","petal_width"]
    )
    pred = model.predict(sample)
    assert pred[0] in [0, 1, 2]

