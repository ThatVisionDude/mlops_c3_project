# Model Card

## Model Details
The model implements a classifier based on a Support Vector Machine. For the training of the model the scikit-learn default parameters were used.

## Intended Use
The model is an example for the online course on MLOps and not as an operational machine learning model for production.

## Training Data
The training data is the [census income dataset](https://archive.ics.uci.edu/ml/datasets/census+income) containing census data and information on the income of the individuals in two classes (>50k/<=50k$ per year).
The categorical variables are one hot encoded. 

## Evaluation Data
The evaluation data has been obtained through a split of the dataset.

## Metrics
The (admittedly not very impressive) metrics are
Precision: 0.9692307692307692
Recall: 0.15879017013232513
FBeta: 0.2728749323226854


## Ethical Considerations
If the model was functional, it would be possible to predict income based on (publicly available) census data, which could be used for a variety of tasks such as marketing or possibly also criminal purposes. Hence care is advised in its use.

## Caveats and Recommendations
This model should not be used in a production environment as it is not optimized in any way.
