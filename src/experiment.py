import io
import csv
from operator import itemgetter
import sys

import pandas
import numpy
import sklearn.ensemble
import sklearn.cross_validation
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.svm
import sklearn.grid_search
import sklearn.tree
import re

TITLE_PATTERN = re.compile(r", +([^.]*)\. ")
DECK_PATTERN = re.compile(r"\b([A-Z]+)\d*\b")
CABIN_NUMBER_PATTERN = re.compile(r"\b[A-Z]+(\d+)\b")
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

def extract_deck(cabin):
    if not cabin:
        return None
    match = DECK_PATTERN.search(cabin)
    if match:
        return match.group(1)
    print "Failed to match ", cabin
    return None

def extract_cabin_number(cabin):
    if not cabin:
        return None
    numbers = CABIN_NUMBER_PATTERN.findall(cabin)
    if numbers:
        return sum(int(n) for n in numbers) / float(len(numbers))
    return 0


def transform_features(data):
    data = data.drop(["Name", "Cabin", "Embarked", "Ticket", "PassengerId", "FamilySize"], axis=1)
    data.info()
    return data.drop("Survived", axis=1).values, data.Survived.values

def fill_age(data, evaluate=True):
    age_data = data[["Age", "Embarked_C", "Embarked_S", "Embarked_Q", "TitleNum", "DeckNum", "CabinNum", "SexNum", "NamesNum", "SibSp", "Parch", "Pclass"]]
    # age_data = data[["Age", "TitleNum", "Pclass"]]

    age_known = age_data[age_data.Age.notnull()]
    age_unknown = age_data[age_data.Age.isnull()]

    X = age_known.drop("Age", axis=1).values
    y = age_known.Age.values

    reg = sklearn.ensemble.RandomForestRegressor(100, n_jobs=-1, random_state=3, oob_score=True)
    reg.fit(X, y)

    split_iterator = sklearn.cross_validation.ShuffleSplit(y.shape[0], n_iter=10, random_state=4)
    cv_scores = sklearn.cross_validation.cross_val_score(reg, X, y, cv=split_iterator)
    print "[Age] Cross-validation accuracy {:.3f} +/- {:.3f}".format(cv_scores.mean(), cv_scores.std() * 2)

    random_params = {
        "max_features": [None, "sqrt", 0.5, 0.8],
        "min_samples_split": [10, 20, 30, 40]
    }

    reg = sklearn.grid_search.GridSearchCV(reg, random_params, n_jobs=-1, cv=split_iterator, refit=True)
    reg.fit(X, y)
    for test in sorted(reg.grid_scores_, key=itemgetter(1), reverse=True):
        print "[Age] Validation score {:.3f}".format(test.mean_validation_score)
        print "[Age] Params", test.parameters
        print ""


    sys.exit(0)

    predicted = reg.predict(age_unknown.drop("Age", axis=1).values)
    data["AgeFill"] = data.Age
    data.loc[data.AgeFill.isnull(), "AgeFill"] = predicted

def clean_data(data):
    data.Embarked = data.Embarked.fillna(data.Embarked.value_counts().idxmax())
    embarked_dummies = pandas.get_dummies(data.Embarked, "Embarked")
    for col in embarked_dummies.columns:
        data[col] = embarked_dummies[col]

    data["Title"] = data.Name.map(extract_title)
    data["NamesNum"] = data.Name.map(lambda n: len(n.split()))

    data["SexNum"] = data.Sex.factorize()[0]
    data.drop("Sex", axis=1, inplace=True)

    data["FamilySize"] = data.SibSp + data.Parch + 1
    data["FarePerPerson"] = data.Fare / data.FamilySize

    # clean up the Fare column
    fare_by_class_embarked = data.pivot_table(index=["Pclass", "Embarked"], values=["FarePerPerson"], aggfunc=numpy.median)
    data["FarePerPersonFill"] = data.FarePerPerson
    data.loc[data.FarePerPersonFill == 0, "FarePerPersonFill"] = None
    for pclass in data.Pclass.unique():
        for embarkation_point in data.Embarked.unique():
            mask = (data.FarePerPersonFill.isnull()) & (data.Pclass == pclass) & (data.Embarked == embarkation_point)
            data.loc[mask, "FarePerPersonFill"] = float(fare_by_class_embarked.ix[pclass, embarkation_point])
    data["FareFill"] = data.FarePerPersonFill * data.FamilySize
    data.drop(["Fare", "FarePerPerson"], axis=1, inplace=True)

    data["TitleNum"] = data.Title.factorize()[0]
    data.drop("Title", axis=1, inplace=True)

    # deck
    data.loc[data.Cabin.isnull(), "Cabin"] = "U0"
    data["Deck"] = data.Cabin.map(extract_deck)
    data["DeckNum"] = pandas.factorize(data.Deck)[0]
    data.drop(["Deck"], axis=1, inplace=True)

    # front/back of boat
    data["CabinNum"] = data.Cabin.map(extract_cabin_number)

    # clean up the Age column
    fill_age(data)
    data.drop("Age", axis=1, inplace=True)

training_data = pandas.read_csv("../data/train.csv", header=0)
test_data = pandas.read_csv("../data/test.csv", header=0)
all_data = pandas.concat([training_data, test_data])

clean_data(all_data)

training_data = all_data[all_data.Survived.notnull()]
test_data = all_data[all_data.Survived.isnull()]
training_data.info()

training_x, training_y = transform_features(training_data)

print "Classifier with fixed hyperparameters"
classifier = sklearn.ensemble.RandomForestClassifier(100, max_features=None, min_samples_split=20, random_state=13, oob_score=True)

# cross-validate the classifier
split_iterator = sklearn.cross_validation.StratifiedShuffleSplit(training_y, n_iter=10, random_state=4)
cv_scores = sklearn.cross_validation.cross_val_score(classifier, training_x, training_y, cv=split_iterator)
print "Cross-validation min {:.3f}".format(cv_scores.min())
print "Cross-validation accuracy {:.3f} +/- {:.3f}".format(cv_scores.mean(), cv_scores.std() * 2)
# print cv_scores

print "Hyperparameter tuning"
base_classifier = sklearn.ensemble.RandomForestClassifier(100, oob_score=True, random_state=13)
random_params = {
    "max_features": [None, "sqrt", 0.5, training_x.shape[1]-1, training_x.shape[1]-2, training_x.shape[1]-3, training_x.shape[1]-4],
    "min_samples_split": [20, 30, 40, 50, 60, 70, 80],
    "min_samples_leaf": [1, 2]
}
tuned_classifier = sklearn.grid_search.GridSearchCV(base_classifier, random_params, n_jobs=-1, cv=split_iterator, refit=True)
tuned_classifier.fit(training_x, training_y)
for test in sorted(tuned_classifier.grid_scores_, key=itemgetter(1), reverse=True):
    print "Validation score {:.3f}".format(test.mean_validation_score)
    print "Params", test.parameters
    print ""

# train the classifier
# tuned_classifier.fit(training_x, training_y)
training_predictions = tuned_classifier.predict(training_x)
diffs = training_predictions - training_y
print "Training accuracy: {:.3f}".format(1. - numpy.abs(diffs).mean())

ids = test_data.PassengerId.values
test_x, _ = transform_features(test_data)
test_predictions = tuned_classifier.predict(test_x)

with io.open("../data/forest_current.csv", "wb") as csv_out:
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(["PassengerId", "Survived"])
    csv_writer.writerows(zip(ids, test_predictions.astype(int)))

