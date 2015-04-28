import io
import csv

import pandas
import numpy
import sklearn.ensemble
import sklearn.cross_validation
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree
import re

TITLE_PATTERN = re.compile(r", +([^.]*)\. ")
TITLE_REMAP = {
    "Capt": "Military",
    "Dona": "Lady",  # Sp/Por female noble
    "Jonkheer": "Sir",  # Dutch lowest title of male nobility
    "Don": "Sir",
    "the Countess": "Lady",
    "Mme": "Mrs",
    "Ms": "Miss",
    "Mlle": "Miss",
    "Major": "Military",
    "Col": "Military"
}


def extract_title(name):
    match = TITLE_PATTERN.search(name)
    if match:
        title = match.group(1)
        return TITLE_REMAP.get(title, title)
    return None


def transform_features(data, age_by_title, age_by_pclass_title, fare_pivot_table):
    data["SexNum"] = data.Sex.factorize()[0]
    data.Embarked = data.Embarked.fillna(data.Embarked.value_counts().idxmax())

    # embarked_dummies = pandas.get_dummies(data.Embarked, "Embarked")
    # data = pandas.concat([data, embarked_dummies], axis=1)

    data["Title"] = data.Name.map(extract_title)
    data["AgeFill"] = data["Age"]

    for title in data.Title.unique():
        for pass_class in data.Pclass.unique():
            mask = (data.Age.isnull()) & (data.Title == title) & (data.Pclass == pass_class)
            try:
                data.loc[mask, "AgeFill"] = float(age_by_pclass_title.ix[pass_class, title])
            except KeyError:
                data.loc[mask, "AgeFill"] = float(age_by_title.ix[title])
    data = data.drop(["Title"], axis=1)

    data["FareFill"] = data["Fare"]
    data.loc[data.FareFill == 0, "FareFill"] = None
    for pclass in data.Pclass.unique():
        for embarkation_point in data.Embarked.unique():
            mask = (data.FareFill.isnull()) & (data.Pclass == pclass) & (data.Embarked == embarkation_point)
            data.loc[mask, "FareFill"] = float(fare_pivot_table.ix[pclass, embarkation_point])
    data = data.drop(["Fare"], axis=1)

    # deck
    # data["Deck"] = data.Cabin.str[0:1]
    # data["DeckNum"] = pandas.factorize(data.Deck)[0] # This might be dangerous cause it'll factorize differently per set
    # data = data.drop(["Deck"], axis=1)

    # data["AgeIsNull"] = data.Age.isnull().astype(int)
    # data["FamilySize"] = data.SibSp + data.Parch
    # data["Age*Class"] = data.AgeFill * data.Pclass


    data = data.drop(["Name", "Cabin", "Age", "Embarked", "Sex", "Ticket", "PassengerId", "SibSp", "Parch"], axis=1)
    # Sex, Class, Fare
    return data.dropna()


training_data = pandas.read_csv("../data/train.csv", header=0)
test_data = pandas.read_csv("../data/test.csv", header=0)
all_data = pandas.concat([training_data, test_data])
all_data["Title"] = all_data.Name.map(extract_title)

age_by_title = all_data.pivot_table(index=["Title"], values=["Age"], aggfunc=numpy.median)
age_by_pclass_title = all_data.pivot_table(index=["Pclass", "Title"], values=["Age"], aggfunc=numpy.median)
fare_table = training_data.pivot_table(index=["Pclass", "Embarked"], values=["Fare"], aggfunc=numpy.median)

training_data = transform_features(training_data, age_by_title, age_by_pclass_title, fare_table)
# print training_data.info()
training_data_values = training_data.values

classifier = sklearn.ensemble.RandomForestClassifier(100, max_features=None, min_samples_split=40, random_state=13)

# cross-validate the classifier
split_iterator = sklearn.cross_validation.StratifiedShuffleSplit(training_data_values[:, 0], n_iter=10, random_state=4)
cv_scores = sklearn.cross_validation.cross_val_score(classifier, training_data_values[:, 1:],
                                                     training_data_values[:, 0], cv=split_iterator)
print "Cross-validation min {:.3f}".format(cv_scores.min())
print "Cross-validation accuracy {:.3f} +/- {:.3f}".format(cv_scores.mean(), cv_scores.std() * 2)
# print cv_scores

# train the classifier
classifier.fit(training_data_values[:, 1:], training_data_values[:, 0])
training_predictions = classifier.predict(training_data_values[:, 1:])
diffs = training_predictions - training_data_values[:, 0]
print "Training accuracy: {:.3f}".format(1. - numpy.abs(diffs).mean())

ids = test_data.PassengerId.values
test_data = transform_features(test_data, age_by_title, age_by_pclass_title, fare_table)

test_predictions = classifier.predict(test_data.values)

with io.open("../data/forest_current.csv", "wb") as csv_out:
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(["PassengerId", "Survived"])
    csv_writer.writerows(zip(ids, test_predictions.astype(int)))