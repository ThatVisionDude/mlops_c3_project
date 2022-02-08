from ml.model import train_model, compute_model_metrics, inference, save_model, load_model
from ml.data import process_data
import pandas as pd
from sklearn.model_selection import train_test_split
import pytest
import sklearn


def test_process_data():
    data = pd.read_csv("data/census_nospace.csv")
    train, test = train_test_split(data, test_size=0.20)
    with pytest.raises(TypeError):
        process_data()
    with pytest.raises(ValueError):
        process_data(train)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=["workclass",
                                     "education",
                                     "marital-status",
                                     "occupation",
                                     "relationship",
                                     "race",
                                     "sex",
                                     "native-country"], label="salary", training=True
    )
    assert len(X_train) == len(train)
    assert len(y_train) == len(train["salary"])


def test_train_model():
    data = pd.read_csv("data/census_nospace.csv")
    data = data[1:100]

    X, y, encoder, lb = process_data(
        data, categorical_features=["workclass",
                                    "education",
                                    "marital-status",
                                    "occupation",
                                    "relationship",
                                    "race",
                                    "sex",
                                    "native-country"], label="salary", training=True
    )

    with pytest.raises(TypeError):
        train_model()
    with pytest.raises(TypeError):
        train_model(X)
    model = train_model(X, y)
    assert isinstance(model, type(sklearn.ensemble.RandomForestClassifier()))


def test_save_load_model():
    data = pd.read_csv("data/census_nospace.csv")
    data = data[1:100]

    X, y, encoder, lb = process_data(
        data, categorical_features=["workclass",
                                    "education",
                                    "marital-status",
                                    "occupation",
                                    "relationship",
                                    "race",
                                    "sex",
                                    "native-country"], label="salary", training=True
    )

    with pytest.raises(TypeError):
        train_model()
    with pytest.raises(TypeError):
        train_model(X)
    model = train_model(X, y)

    with pytest.raises(TypeError):
        save_model()
    with pytest.raises(TypeError):
        load_model()

    with pytest.raises(TypeError):
        save_model(model, 1)

    with pytest.raises(AssertionError):
        save_model(model, encoder, lb,  1)
    with pytest.raises(AssertionError):
        load_model(1)

    save_model(model, encoder, lb, "testfile")
    modelLoaded, encLoaded, lbLoaded = load_model("testfile")

    assert(isinstance(model, type(modelLoaded)))
    assert((inference(model, X) == inference(modelLoaded, X)).all())
