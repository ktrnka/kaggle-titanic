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

sex_map = {"female": 1, "male": 0}


def transform_features(data, ages):
    data["SexNum"] = data.Sex.map(sex_map).astype(int)
    # data["EmbarkedS"] = (data.Embarked == "S").astype(int)
    # data["EmbarkedC"] = (data.Embarked == "C").astype(int)
    # data["EmbarkedQ"] = (data.Embarked == "Q").astype(int)

    data["AgeFill"] = data["Age"]

    for sex in xrange(2):
        for pass_class in xrange(3):
            mask = (data.Age.isnull()) & (data.SexNum == sex) & (data.Pclass == pass_class + 1)
            data.loc[mask, "AgeFill"] = ages[sex, pass_class]

    #data["AgeIsNull"] = data.Age.isnull().astype(int)
    # data["FamilySize"] = data.SibSp + data.Parch
    #data["Age*Class"] = data.AgeFill * data.Pclass

    data.loc[data.Fare.isnull(), "Fare"] = data.Fare.dropna().median()

    data = data.drop(["Name", "Cabin", "Age", "Embarked", "Sex", "Ticket", "PassengerId", "SibSp", "Parch"], axis=1)
    # Sex, Class, Fare
    return data.dropna()


def build_age_matrix(data):
    ages = numpy.zeros((2, 3))
    reverse_sex_map = {val: key for key, val in sex_map.iteritems()}
    for sex in xrange(2):
        for pass_class in xrange(3):
            mask = (data.Sex == reverse_sex_map[sex]) & (data.Pclass == pass_class + 1)
            ages[sex, pass_class] = data[mask].Age.dropna().median()
    return ages


data = pandas.read_csv("../data/train.csv", header=0)

ages = build_age_matrix(data)

data = transform_features(data, ages)
training_data = data.values

classifier = sklearn.ensemble.RandomForestClassifier(100, max_features=None, min_samples_leaf=3, random_state=13)
#classifier = sklearn.linear_model.LogisticRegression(C=10.)

# cross-validate the classifier
split_iterator = sklearn.cross_validation.StratifiedShuffleSplit(training_data[:, 0], n_iter=10, random_state=4)
cv_scores = sklearn.cross_validation.cross_val_score(classifier, training_data[:, 1:], training_data[:, 0], cv=split_iterator)
print "Cross-validation min {:.3f}".format(cv_scores.min())
print "Cross-validation accuracy {:.3f} +/- {:.3f}".format(cv_scores.mean(), cv_scores.std() * 2)
# print cv_scores

# train the classifier
classifier.fit(training_data[:, 1:], training_data[:, 0])
training_predictions = classifier.predict(training_data[:, 1:])
diffs = training_predictions - training_data[:, 0]
print "Training accuracy: {:.3f}".format(1. - numpy.abs(diffs).mean())

print data.info()
# print "Weights", zip(data.columns[1:].values, classifier.coef_[0,:])


test_data = pandas.read_csv("../data/test.csv", header=0)
ids = test_data.PassengerId.values
test_data = transform_features(test_data, ages)

test_predictions = classifier.predict(test_data.values)

with io.open("../data/forest_current.csv", "wb") as csv_out:
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(["PassengerId", "Survived"])
    csv_writer.writerows(zip(ids, test_predictions.astype(int)))