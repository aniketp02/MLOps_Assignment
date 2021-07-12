import dvc.api
import pandas as pd

with dvc.api.open(repo = "https://github.com/aniketp02/MLOps_Assignment", path = "data/creditcard.csv") as fd:
    df = pd.read_csv(fd)


# to split the data into 80:20
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2, random_state=3)


# saving the data in the processed folder as train.csv and test.csv
folder = '../data/prepared'

train.to_csv(folder + '/train.csv', index=False)
test.to_csv(folder + '/test.csv', index=False)


# training a decision tree classifier
x_train = train.copy().drop('Class', axis=1)
y_train = train['Class']

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=3)
clf.fit(x_train, y_train)


# saving the trained model in models as .pkl
import pickle
pickle.dump(clf, open('../models/model.pkl', 'wb'))


# evaluating model on test data
x_test = test.copy().drop('Class', axis=1)
y_test = test['Class']

pkl_model = pickle.load(open('../models/model.pkl', 'rb'))
y_pred = pkl_model.predict(x_test)


# saving results to a JSON file
from sklearn.metrics import accuracy_score, f1_score
import json

results = {
    'Accuracy' : accuracy_score(y_test, y_pred),
    'Weighted F1 score' : f1_score(y_test, y_pred, average='weighted')
}

json.dump(results, open('../metrics/acc_f1.json', 'w'))