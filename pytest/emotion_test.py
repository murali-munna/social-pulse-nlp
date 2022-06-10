from pathlib import Path
import pytest
import numpy as np
from config import config
from tagifai import main, predict
from Emotional_model import *

def test_generate():
    train_X, test_X, train_y, test_y = generate_train_test("training.csv"):
    assert isinstance(train_X, np.array)
    assert isinstance(train_y, np.array)
    assert isinstance(test_X, np.array)
    assert isinstance(test_y, np.array)
    with pytest.raises(ValueError):
        train_X = []
        train_y = []
        test_X = []
        test_y = []
        
def test_train():
    model = train_test("training")
    assert isinstance(train_test.pred, np.array)
    with pytest.raises(ValueError):
        train_test.pred = []
        train_test(data_set = None)
        
def test_pred():
    model = train_test("training")
    pred = single_predict(model, "How are you?")
    pred_2 = single_predict(model, "testing.csv")
    assert isinstance(pred, int)
    assert isinstance(pred_2, list)
    with pytest.raises(ValueError):
        pred = []
        pred_2 = []
        
def test_clean():
    cleaned_text = clean_text("#!_&text?")
    assert isinstance(cleaned_text, str)
    assert cleaned_text == 'text'

def artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    return artifacts

@pytest.mark.parametrize(
    "text, tag",
    [
        ("Transformers applied to NLP have revolutionized machine learning.", "natural-language-processing"),
        ("Transformers applied to NLP have disrupted machine learning.", "natural-language-processing"),
    ],
)
def test_inv(text, tag, artifacts):
    """INVariance via verb injection (changes should not affect outputs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
    assert tag == predicted_tag

@pytest.mark.parametrize(
    "text, tag",
    [
        ("ML applied to text classification.", "natural-language-processing"),
        ("ML applied to image classification.", "computer-vision"),
    ],
)
def test_dir(text, tag, artifacts):
    """DIRectional expectations (changes with known outputs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
    assert tag == predicted_tag

@pytest.mark.parametrize(
    "text, tag",
    [
        ("Natural language processing is the next big wave in machine learning..", "natural-language-processing"),
        ("MLOps is the next big wave in machine learning..", "mlops"),
    ],
)
def test_mft(text, tag, artifacts):
    """Minimum Functionality Tests (simple input/output pairs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
    assert tag == predicted_tag
