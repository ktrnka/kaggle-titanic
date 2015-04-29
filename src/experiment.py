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


def transform_features(data):
    data = data.drop(["Name", "Cabin", "Embarked", "Ticket", "PassengerId", "FamilySize"], axis=1)
    data.info()
    return data.drop("Survived", axis=1).values, data.Survived.values

def clean_data(data):
    data.Embarked = data.Embarked.fillna(data.Embarked.value_counts().idxmax())
    embarked_dummies = pandas.get_dummies(data.Embarked, "Embarked")
    for col in embarked_dummies.columns:
        data[col] = embarked_dummies[col]

    data["Title"] = data.Name.map(extract_title)

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
    data.drop(["Fare", "FarePerPerson", "FarePerPersonFill"], axis=1, inplace=True)

    # clean up the Age column
    age_by_title = data.pivot_table(index=["Title"], values=["Age"], aggfunc=numpy.median)
    age_by_pclass_title = data.pivot_table(index=["Pclass", "Title"], values=["Age"], aggfunc=numpy.median)

    data["AgeFill"] = data["Age"]
    for title in data.Title.unique():
        for pclass in data.Pclass.unique():
            mask = (data.Age.isnull()) & (data.Title == title) & (data.Pclass == pclass)
            try:
                data.loc[mask, "AgeFill"] = float(age_by_pclass_title.ix[pclass, title])
            except KeyError:
                data.loc[mask, "AgeFill"] = float(age_by_title.ix[title])
    data.drop("Age", axis=1, inplace=True)

    data["TitleNum"] = data.Title.factorize()[0]
    data.drop("Title", axis=1, inplace=True)

    # deck
    # data["Deck"] = data.Cabin.str[0:1]
    # data["DeckNum"] = pandas.factorize(data.Deck)[0] # This might be dangerous cause it'll factorize differently per set
    # data = data.drop(["Deck"], axis=1)


training_data = pandas.read_csv("../data/train.csv", header=0)
test_data = pandas.read_csv("../data/test.csv", header=0)
all_data = pandas.concat([training_data, test_data])

clean_data(all_data)

training_data = all_data[all_data.Survived.notnull()]
test_data = all_data[all_data.Survived.isnull()]
training_data.info()

training_x, training_y = transform_features(training_data)

classifier = sklearn.ensemble.RandomForestClassifier(100, max_features=None, min_samples_split=20, random_state=13, oob_score=True)

# cross-validate the classifier
split_iterator = sklearn.cross_validation.StratifiedShuffleSplit(training_y, n_iter=10, random_state=4)
cv_scores = sklearn.cross_validation.cross_val_score(classifier, training_x, training_y, cv=split_iterator)
print "Cross-validation min {:.3f}".format(cv_scores.min())
print "Cross-validation accuracy {:.3f} +/- {:.3f}".format(cv_scores.mean(), cv_scores.std() * 2)
# print cv_scores

# train the classifier
classifier.fit(training_x, training_y)
training_predictions = classifier.predict(training_x)
diffs = training_predictions - training_y
print "Training accuracy: {:.3f}".format(1. - numpy.abs(diffs).mean())

ids = test_data.PassengerId.values
test_x, _ = transform_features(test_data)
test_predictions = classifier.predict(test_x)

with io.open("../data/forest_current.csv", "wb") as csv_out:
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(["PassengerId", "Survived"])
    csv_writer.writerows(zip(ids, test_predictions.astype(int)))