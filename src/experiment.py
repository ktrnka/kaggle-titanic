import io
import csv
from operator import itemgetter
import pprint
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
import sklearn.learning_curve
import matplotlib.pyplot as plt
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
DEFAULT_DECK = "U"


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

def extract_ticket_number_part(ticket):
    if not ticket:
        return -1

    parts = ticket.split()

    try:
        return int(parts[-1])
    except ValueError:
        print parts[-1]
        return -1

def extract_ticket_alpha_part(ticket):
    if not ticket:
        return None

    parts = ticket.split()

    try:
        num = int(parts[0])
        return None
    except ValueError:
        return parts[0]

def transform_features(data):
    data = data.drop(["Name", "Cabin", "Embarked", "Ticket", "PassengerId", "FamilySize"], axis=1)
    data.info()
    print "Data columns: {}".format(", ".join(sorted(data.columns)))
    X = data.drop("Survived", axis=1)
    return X.values, data.Survived.values, X.columns


def print_tuning_scores(tuned_estimator, reverse=True):
    for test in sorted(tuned_estimator.grid_scores_, key=itemgetter(1), reverse=reverse):
        print "Validation score {:.3f}, Hyperparams {}".format(test.mean_validation_score, test.parameters)


def fill_age(data, evaluate=True):
    # age and the features used to predict it
    age_data = data[
        ["Age", "Embarked_C", "Embarked_S", "Embarked_Q", "TitleNum", "DeckNum", "CabinNum", "SexNum", "NamesNum",
         "SibSp", "Parch", "Pclass"]]

    age_known = age_data[age_data.Age.notnull()]
    age_unknown = age_data[age_data.Age.isnull()]

    X = age_known.drop("Age", axis=1).values
    y = age_known.Age.values

    regressor = sklearn.ensemble.RandomForestRegressor(100, n_jobs=-1, random_state=3, oob_score=True)

    split_iterator = sklearn.cross_validation.ShuffleSplit(y.shape[0], n_iter=10, random_state=4)

    if evaluate:
        cv_scores = sklearn.cross_validation.cross_val_score(regressor, X, y, cv=split_iterator)
        print "[Age] Cross-validation accuracy {:.3f} +/- {:.3f}".format(cv_scores.mean(), cv_scores.std() * 2)

    hyperparams = {
        "max_features": [None, "sqrt", 0.5, 0.8],
        "min_samples_split": [10, 20, 30, 40]
    }

    regressor_tuning = sklearn.grid_search.GridSearchCV(regressor, hyperparams, n_jobs=-1, cv=split_iterator,
                                                        refit=True)
    regressor_tuning.fit(X, y)

    if evaluate:
        print "Age hyperparameter tuning"
        paired_features = zip(age_known.drop("Age", axis=1).columns, regressor_tuning.best_estimator_.feature_importances_)
        pprint.pprint(sorted(paired_features, key=itemgetter(1), reverse=True))

        print_tuning_scores(regressor_tuning)

    predicted = regressor_tuning.predict(age_unknown.drop("Age", axis=1).values)
    data["AgeFill"] = data.Age
    data.loc[data.AgeFill.isnull(), "AgeFill"] = predicted

def fill_fare_per_person(data, evaluate=True):
    # fare and predictors
    fare_data = data[["FarePerPerson", "Pclass", "Embarked_Q", "Embarked_S", "Embarked_C", "TitleNum", "SexNum", "DeckNum", "CabinNum", "SibSp", "Parch"]]

    fare_known = fare_data[fare_data.FarePerPerson.notnull()]
    fare_unknown = fare_data[fare_data.FarePerPerson.isnull()]

    X = fare_known.drop("FarePerPerson", axis=1).values
    y = fare_known.FarePerPerson.values

    regressor = sklearn.ensemble.RandomForestRegressor(100, n_jobs=-1, random_state=3, oob_score=True)

    split_iterator = sklearn.cross_validation.ShuffleSplit(y.shape[0], n_iter=10, random_state=4)

    if evaluate:
        cv_scores = sklearn.cross_validation.cross_val_score(regressor, X, y, cv=split_iterator)
        print "[FarePerPerson] Cross-validation accuracy {:.3f} +/- {:.3f}".format(cv_scores.mean(), cv_scores.std() * 2)

    hyperparams = {
        "max_features": [None, "sqrt", 0.5, 0.8],
        "min_samples_split": [10, 20]
    }

    regressor_tuning = sklearn.grid_search.GridSearchCV(regressor, hyperparams, n_jobs=-1, cv=split_iterator,
                                                        refit=True)
    regressor_tuning.fit(X, y)

    if evaluate:
        print "FarePerPerson hyperparameter tuning"
        paired_features = zip(fare_known.drop("FarePerPerson", axis=1).columns, regressor_tuning.best_estimator_.feature_importances_)
        pprint.pprint(sorted(paired_features, key=itemgetter(1), reverse=True))
        print_tuning_scores(regressor_tuning)

    predicted = regressor_tuning.predict(fare_unknown.drop("FarePerPerson", axis=1).values)
    data["FarePerPersonFill"] = data.FarePerPerson
    data.loc[data.FarePerPersonFill.isnull(), "FarePerPersonFill"] = predicted

    # sys.exit(0)


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
    data.loc[data.FarePerPerson == 0, "FarePerPerson"] = None
    # data["FareMissing"] = data.FarePerPersonFill.isnull().astype(int)
    for pclass in data.Pclass.unique():
        for embarkation_point in data.Embarked.unique():
            mask = (data.FarePerPersonFill.isnull()) & (data.Pclass == pclass) & (data.Embarked == embarkation_point)
            data.loc[mask, "FarePerPersonFill"] = float(fare_by_class_embarked.ix[pclass, embarkation_point])

    data["TitleNum"] = data.Title.factorize()[0]
    title_indicators = pandas.get_dummies(data.Title, "Title")
    for column in title_indicators.columns:
        data[column] = title_indicators[column]
    data.drop("Title", axis=1, inplace=True)

    # deck
    # data["CabinKnown"] = data.Cabin.notnull().astype(int)
    data.loc[data.Cabin.isnull(), "Cabin"] = DEFAULT_DECK + "0"
    data["Deck"] = data.Cabin.map(extract_deck)
    # deck_dummies = pandas.get_dummies(data.Deck, "Deck")
    # for col in deck_dummies.columns:
    #     data[col] = deck_dummies[col]

    data["DeckNum"] = pandas.factorize(data.Deck)[0]
    data.drop(["Deck"], axis=1, inplace=True)

    # front/back of boat
    data["CabinNum"] = data.Cabin.map(extract_cabin_number)
    data["ShipSide"] = numpy.round(data.CabinNum) % 2
    data.loc[data.CabinNum == 0, "ShipSide"] = -1

    # clean up the fare
    fill_fare_per_person(data)
    data["FareFill"] = data.FarePerPersonFill * data.FamilySize
    data.drop(["Fare", "FarePerPerson"], axis=1, inplace=True)

    # clean up the Age column
    # data["AgeMissing"] = data.Age.isnull().astype(int)
    fill_age(data)
    data.drop("Age", axis=1, inplace=True)

    data["TicketAlphaPart"] = pandas.factorize(data.Ticket.map(extract_ticket_alpha_part).str.upper().str.replace(r"\.", ""))[0]
    data["TicketNumPart"] = data.Ticket.map(extract_ticket_number_part)
    # data["TicketNumPart"] -= data["TicketNumPart"] % 100

    # # ticket feature
    # data["TicketNum"] = -1
    # data["GroupSize"] = 1
    # shared_tickets = dict(data.Ticket.value_counts())
    # shared_tickets = {ticket: count for ticket, count in shared_tickets.iteritems() if count > 1}
    # for i, ticket in enumerate(shared_tickets.iterkeys()):
    #     data.loc[data.Ticket == ticket, "TicketNum"] = i
    #     data.loc[data.Ticket == ticket, "GroupSize"] = shared_tickets[ticket]

    # data.drop(["DeckNum"], axis=1, inplace=True)

    # create some binned indexed versions
    # for col, bins in [("FarePerPersonFill", 5), ("FareFill", 5), ("AgeFill", 10)]:
    #     binned_data = pandas.qcut(data[col], bins)
    #     data[col + "_bin"] = pandas.factorize(binned_data, sort=True)[0]

    # drop from temp vars
    data.drop(["TitleNum"], axis=1, inplace=True)


def learning_curve(training_x, training_y, filename):
    split_iterator = sklearn.cross_validation.StratifiedShuffleSplit(training_y, min_samples_split=8, min_samples_leaf=2, n_iter=10, random_state=4)
    base_classifier = sklearn.ensemble.RandomForestClassifier(100, random_state=13)
    train_sizes, train_scores, test_scores = sklearn.learning_curve.learning_curve(base_classifier, training_x,
                                                                                   training_y, cv=split_iterator,
                                                                                   train_sizes=numpy.linspace(.1, 1., 10),
                                                                                   verbose=0)

    training_means = train_scores.mean(axis=1)
    training_std = train_scores.std(axis=1)
    test_means = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    plt.figure()
    plt.title("Random Forest Classifier")
    plt.legend(loc="best")
    plt.xlabel("Training size")
    plt.ylabel("Accuracy")
    plt.ylim((0.6, 1.01))
    plt.gca().invert_yaxis()
    plt.grid()

    plt.plot(train_sizes, training_means, "o-", color="b", label="Training")
    plt.plot(train_sizes, test_means, "o-", color="r", label="Testing")

    plt.fill_between(train_sizes, training_means - training_std, training_means + training_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_means - test_std, test_means + test_std, alpha=0.1, color="r")

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def main():
    training_data = pandas.read_csv("../data/train.csv", header=0)
    test_data = pandas.read_csv("../data/test.csv", header=0)
    all_data = pandas.concat([training_data, test_data])

    clean_data(all_data)

    training_data = all_data[all_data.Survived.notnull()]
    test_data = all_data[all_data.Survived.isnull()]

    training_x, training_y, columns = transform_features(training_data)

    # cross-validate the classifier
    split_iterator = sklearn.cross_validation.StratifiedShuffleSplit(training_y, n_iter=10, random_state=4)

    # learning curve with default settings
    learning_curve(training_x, training_y, "learning_curve.png")

    print "Hyperparameter tuning"
    base_classifier = sklearn.ensemble.RandomForestClassifier(100, oob_score=True, random_state=13)
    parameter_space = {
        "max_features": [None, "sqrt", 0.5, training_x.shape[1] - 1, training_x.shape[1] - 2, training_x.shape[1] - 3,
                         training_x.shape[1] - 4],
        "min_samples_split": [20, 30, 40],
        "min_samples_leaf": [1, 2, 3, 4]
    }
    tuned_classifier = sklearn.grid_search.GridSearchCV(base_classifier, parameter_space, n_jobs=-1, cv=split_iterator, refit=True)
    tuned_classifier.fit(training_x, training_y)
    print_tuning_scores(tuned_classifier)

    paired_features = zip(columns, tuned_classifier.best_estimator_.feature_importances_)
    pprint.pprint(sorted(paired_features, key=itemgetter(1), reverse=True))

    training_predictions = tuned_classifier.predict(training_x)
    diffs = training_predictions - training_y
    print "Training accuracy: {:.3f}".format(1. - numpy.abs(diffs).mean())

    ids = test_data.PassengerId.values
    test_x, _, _ = transform_features(test_data)
    test_predictions = tuned_classifier.predict(test_x)

    with io.open("../data/forest_current.csv", "wb") as csv_out:
        csv_writer = csv.writer(csv_out)
        csv_writer.writerow(["PassengerId", "Survived"])
        csv_writer.writerows(zip(ids, test_predictions.astype(int)))


if __name__ == "__main__":
    sys.exit(main())

