from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

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
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf = 2)
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

def save_model(model, encoder: OneHotEncoder, label_binarizer: LabelBinarizer, name: str):
    assert(type(name)==str)
    from joblib import dump
    print("Saving model to file " + name)
    dump(model, name)
    print("Saving encoder to file " + name + '_enc')
    dump(encoder, name+ '_enc')
    print("Saving label binarizer to file " + name + '_lb')
    dump(label_binarizer, name + '_lb')

def load_model(name: str):
    from joblib import load
    assert(type(name)==str)
    print("Loading model from file: " + name)
    model = load(name)
    print("Loading encoder from file " + name + '_enc')
    enc = load(name+ '_enc')
    print("Loading label binarizer from file " + name + '_lb')
    lb = load(name + '_lb')
    return [model, enc, lb]

def get_rows(data, cat_variable, unique_cls):
    return data[data[cat_variable] == unique_cls].index

def slice_performance(original_data, X,y, model, cat_variable, filename = 'slice_output.txt'):
    """ 
    Slice the data along a categorical variable and 
    output the performance of the model
    """
    with open(filename, 'a') as fn:
        fn.write(f"Checking categorical variable {cat_variable} \n")
    
    for unique_cls in original_data[cat_variable].unique():
        idx = get_rows(original_data.reset_index(), cat_variable, unique_cls)
        X_cls = X[idx,:]
        y_cls = y[idx]
        
        precision, recall, fbeta = compute_model_metrics(y_cls, inference(model, X_cls))
        with open(filename, 'a') as fn:    
            fn.write(f"Class: {unique_cls} \n")
            fn.write("Precision: " + str(precision) +"\n")
            fn.write("Recall: " + str(recall)+"\n")
            fn.write("FBeta: " + str(fbeta)+"\n")
