import io
import csv

import pandas
import numpy
import sklearn.ensemble
import sklearn.cross_validation


data = pandas.read_csv("../data/train.csv", header=0)
data["SexNum"] = data.Sex.map({"female": 1, "male": 0}).astype(int)
data["EmbarkedNum"] = data.Embarked.map({"S": 0, "C": 1, "Q": 2, None: 3}).astype(int)

data["AgeFill"] = data["Age"]

ages = numpy.zeros((2, 3))
for sex in xrange(2):
    for pass_class in xrange(3):
        ages[sex, pass_class] = data[(data.SexNum == sex) & (data.Pclass == pass_class+1)].Age.dropna().median()
        data.loc[(data.Age.isnull()) & (data.SexNum == sex) & (data.Pclass == pass_class+1),"AgeFill"] = ages[sex, pass_class]

#data["AgeIsNull"] = data.Age.isnull().astype(int)
data["FamilySize"] = data.SibSp + data.Parch
#data["Age*Class"] = data.AgeFill * data.Pclass

data = data.drop(["Name", "Cabin", "Age", "Embarked", "Sex", "Ticket", "PassengerId"], axis=1)
data = data.dropna()

training_data = data.values

classifier = sklearn.ensemble.RandomForestClassifier(100, oob_score=True, max_features=0.5, min_samples_split=50)

# cross-validate the classifier
cv_scores = sklearn.cross_validation.cross_val_score(classifier, training_data[:,1:], training_data[:,0], cv=10)
print "Cross-validation accuracy {:.3f} +/- {:.3f}".format(cv_scores.mean(), cv_scores.std() * 2)
print cv_scores

# train the classifier
classifier.fit(training_data[:,1:], training_data[:,0])
training_predictions = classifier.predict(training_data[:,1:])
diffs = training_predictions - training_data[:,0]
print "Training error", numpy.abs(diffs).mean()


test_data = pandas.read_csv("../data/test.csv", header=0)

test_data["SexNum"] = test_data.Sex.map({"female": 1, "male": 0}).astype(int)
test_data["EmbarkedNum"] = test_data.Embarked.map({"S": 0, "C": 1, "Q": 2, None: 3}).astype(int)

test_data["AgeFill"] = test_data["Age"]
for sex in xrange(2):
    for pass_class in xrange(3):
        test_data.loc[(test_data.Age.isnull()) & (test_data.SexNum == sex) & (test_data.Pclass == pass_class+1),"AgeFill"] = ages[sex, pass_class]

#test_data["AgeIsNull"] = test_data.Age.isnull().astype(int)
test_data["FamilySize"] = test_data.SibSp + test_data.Parch
#test_data["Age*Class"] = test_data.AgeFill * test_data.Pclass
test_data.loc[test_data.Fare.isnull(), "Fare"] = test_data.Fare.dropna().median()

ids = test_data.PassengerId.values
test_data = test_data.drop(["Name", "Cabin", "Age", "Embarked", "Sex", "Ticket", "PassengerId"], axis=1)
# test_data.info()

test_predictions = classifier.predict(test_data.values)

with io.open("../data/forest3.csv", "wb") as csv_out:
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(["PassengerId", "Survived"])
    csv_writer.writerows(zip(ids, test_predictions.astype(int)))