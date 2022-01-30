from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn import svm
from sklearn.metrics import accuracy_score

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise Exception("X_train and y_train must have the same shape.")
    model = svm.SVC()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Support vector classifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def save_model(model, name: str):
    assert(type(name)==str)
    from joblib import dump
    print("Saving model to file " + name)
    dump(model, name)

def load_model(name: str):
    from joblib import load
    assert(type(name)==str)
    print("Loading model from file: " + name)
    model = load(name)
    return model