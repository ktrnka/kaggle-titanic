import io
import csv

import pandas
import numpy
import sklearn.ensemble
import sklearn.cross_validation
import sklearn.linear_model

sex_map = {"female": 1, "male": 0}

def transform_features(data, ages):
    data["SexNum"] = data.Sex.map(sex_map).astype(int)
    data["EmbarkedS"] = (data.Embarked == "S").astype(int)
    data["EmbarkedC"] = (data.Embarked == "C").astype(int)
    data["EmbarkedQ"] = (data.Embarked == "Q").astype(int)

    data["AgeFill"] = data["Age"]

    for sex in xrange(2):
        for pass_class in xrange(3):
            data.loc[(data.Age.isnull()) & (data.SexNum == sex) & (data.Pclass == pass_class + 1), "AgeFill"] = ages[sex, pass_class]

    # data["AgeIsNull"] = data.Age.isnull().astype(int)
    data["FamilySize"] = data.SibSp + data.Parch
    #data["Age*Class"] = data.AgeFill * data.Pclass

    data.loc[data.Fare.isnull(), "Fare"] = data.Fare.dropna().median()

    data = data.drop(["Name", "Cabin", "Age", "Embarked", "Sex", "Ticket", "PassengerId"], axis=1)
    return data.dropna()


def build_age_matrix(data):
    ages = numpy.zeros((2, 3))
    reverse_sex_map = {val: key for key, val in sex_map.iteritems()}
    for sex in xrange(2):
        for pass_class in xrange(3):
            ages[sex, pass_class] = data[(data.Sex == reverse_sex_map[sex]) & (data.Pclass == pass_class + 1)].Age.dropna().median()
            data.loc[(data.Age.isnull()) & (data.Sex == reverse_sex_map[sex]) & (data.Pclass == pass_class + 1), "AgeFill"] = ages[sex, pass_class]
    return ages


data = pandas.read_csv("../data/train.csv", header=0)

ages = build_age_matrix(data)

data = transform_features(data, ages)
training_data = data.values

# classifier = sklearn.ensemble.RandomForestClassifier(1000, max_features=0.8, min_samples_split=50)
classifier = sklearn.linear_model.LogisticRegression(penalty="l2", C=10.)

# cross-validate the classifier
cv_scores = sklearn.cross_validation.cross_val_score(classifier, training_data[:, 1:], training_data[:, 0], cv=10)
print "Cross-validation accuracy {:.3f} +/- {:.3f}".format(cv_scores.mean(), cv_scores.std() * 2)
print cv_scores

# train the classifier
classifier.fit(training_data[:, 1:], training_data[:, 0])
training_predictions = classifier.predict(training_data[:, 1:])
diffs = training_predictions - training_data[:, 0]
print "Training error", numpy.abs(diffs).mean()

test_data = pandas.read_csv("../data/test.csv", header=0)
ids = test_data.PassengerId.values
test_data = transform_features(test_data, ages)

test_predictions = classifier.predict(test_data.values)

with io.open("../data/forest3.csv", "wb") as csv_out:
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(["PassengerId", "Survived"])
    csv_writer.writerows(zip(ids, test_predictions.astype(int)))