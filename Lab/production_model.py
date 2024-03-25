
import pandas as pd
import joblib

test_samples = pd.read_csv("Lab/Data/test_samples.csv", index_col=0)

model = joblib.load('Lab/Data/voting_clf.pkl')

# use trained model to predict on test_samples
X_test = test_samples.drop('cardio', axis=1)
y_test = test_samples['cardio']



y_pred = model.predict_proba(X_test)

print("Accuracy score: ", model.score(X_test, y_test))


predictions = pd.DataFrame(y_pred, columns=['Probability class 0', 'Probability class 1'])
predictions['Prediction'] = model.predict(X_test)

# export the predictions to a csv file
predictions.to_csv('Lab/Data/predictions.csv', index=False)