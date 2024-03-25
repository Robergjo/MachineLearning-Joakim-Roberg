
import pandas as pd
from joblib import load as ld

test_data = pd.read_csv("MachineLearning-Joakim-Roberg\\Lab\\Data\\test_samples.csv", index_col= "id")

model = ld('MachineLearning-Joakim-Roberg\\Lab\\voting_clf.pkl')

X_test = test_data.drop("cardio", axis = 1)

probabilities = model.predict_proba(X_test)

predictions = model.predict(X_test)

results = pd.DataFrame({
    'probability class 0': probabilities[:, 0],
    'probability class 1': probabilities[:, 1],
    'prediction': predictions
})

results.to_csv("MachineLearning-Joakim-Roberg\\Lab\\predictions.csv")