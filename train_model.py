# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, save_model, compute_model_metrics, inference, slice_performance
# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("data/census_nospace.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, training=False, label="salary", 
    categorical_features=cat_features, 
    encoder = encoder, lb = lb)
# Train and save a model.
used_model = train_model(X_train, y_train)
precision, recall, fbeta = compute_model_metrics(y_test,inference(used_model, X_test))
print("Overall accuracy:")
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("FBeta: " + str(fbeta))
save_model(used_model, encoder, lb, "model")

slice_performance(test, X_test, y_test, used_model, "education")
