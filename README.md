Repo for Kaggle Titanic tutorial competition

History
-------
Gender-based probabilities: 0.76555
Gender, Fare bin, Class: 0.77990
Random Forest (tut): 0.73684
RF with 100% of features: 0.74641
RF with 100% of features and more limited trees: 0.74641
RF with 100% features, MSS=50: 0.77033
+ max_depth=2: 0.76555
without max_depth, dropping some features: 0.77033
RF 80% features, MSS=50, dropping a couple features: 0.77512
RF 50% features, MSS=50, dropping a couple features: 0.79904

RF 50%, MSS=50: Cross-validation accuracy 0.827 +/- 0.081
[ 0.8         0.83333333  0.75280899  0.88764045  0.86516854  0.79775281
  0.80898876  0.79775281  0.87640449  0.85227273]
  
max_features=0.8, min_samples_split=50
Cross-validation accuracy 0.824 +/- 0.083
[ 0.76666667  0.83333333  0.76404494  0.88764045  0.87640449  0.80898876
  0.80898876  0.78651685  0.85393258  0.85227273]
Training error 0.148148148148

with 1000 trees:
Cross-validation accuracy 0.817 +/- 0.078
[ 0.76666667  0.81111111  0.76404494  0.88764045  0.86516854  0.80898876
  0.80898876  0.78651685  0.85393258  0.81818182]
Training error 0.149270482604

Logistic Regression, C=1.
Cross-validation accuracy 0.801 +/- 0.056
[ 0.78888889  0.78888889  0.7752809   0.86516854  0.80898876  0.7752809
  0.78651685  0.7752809   0.83146067  0.81818182]
Training error 0.190796857464

C=0.1
Cross-validation accuracy 0.801 +/- 0.063
[ 0.8         0.77777778  0.7752809   0.86516854  0.80898876  0.78651685
  0.78651685  0.76404494  0.85393258  0.79545455]
Training error 0.191919191919

C=10.
Cross-validation accuracy 0.805 +/- 0.047
[ 0.8         0.78888889  0.7752809   0.85393258  0.83146067  0.7752809
  0.79775281  0.79775281  0.82022472  0.80681818]
Training error 0.190796857464

Changing Embarked to booleans
Cross-validation accuracy 0.805 +/- 0.043
[ 0.8         0.78888889  0.78651685  0.85393258  0.80898876  0.7752809
  0.80898876  0.79775281  0.79775281  0.82954545]
Training error 0.191919191919

LR C=1.
0.36842 on test data

LR C=0.1
Cross-validation accuracy 0.798 +/- 0.055
[ 0.78888889  0.78888889  0.7752809   0.85393258  0.79775281  0.7752809
  0.78651685  0.76404494  0.84269663  0.80681818]
Training error 0.190796857464

KR defaults
Cross-validation accuracy 0.800 +/- 0.049
[ 0.78888889  0.78888889  0.7752809   0.85393258  0.79775281  0.7752809
  0.78651685  0.78651685  0.83146067  0.81818182]
Training error 0.194163860831

Basic Logistic Regression
Cross-validation accuracy 0.801 +/- 0.056
[ 0.78888889  0.78888889  0.7752809   0.86516854  0.80898876  0.7752809
  0.78651685  0.7752809   0.83146067  0.81818182]
Training error 0.190796857464

Basic Logistic Regression, dropping the Fare column
Cross-validation accuracy 0.803 +/- 0.038
[ 0.78888889  0.78888889  0.78651685  0.84269663  0.80898876  0.78651685
  0.79775281  0.78651685  0.80898876  0.82954545]
Training error 0.189674523008

Basic LR C = 10 no Fare
0.75598

RandomForestClassifier(1000, max_features=0.8, min_samples_split=50) no fare
0.76077,

classifier = sklearn.ensemble.RandomForestClassifier(100, max_features=None)
gender, Pclass, Fare (no other features)
Cross-validation accuracy 0.817 +/- 0.077
[ 0.77777778  0.77777778  0.78651685  0.78651685  0.91011236  0.80898876
  0.83146067  0.83146067  0.82022472  0.84090909]
Training error 0.0953984287318

Error is 0.76077

Overfitting probably due to the Fare column


Previous experiment with Fare in 10-dollar bins
-----------------------------------------------
Cross-validation accuracy 0.806 +/- 0.056
[ 0.81111111  0.78888889  0.75280899  0.83146067  0.84269663  0.7752809
  0.79775281  0.79775281  0.84269663  0.81818182]
Training error 0.182940516274

Test accuracy is 0.77512 (good improvement but still overfitting)

No fare bins but high min samples split
---------------------------------------
sklearn.ensemble.RandomForestClassifier(100, max_features=None, min_samples_split=80)
Cross-validation min 0.775
Cross-validation accuracy 0.807 +/- 0.044
[ 0.82222222  0.8         0.7752809   0.83146067  0.83146067  0.78651685
  0.79775281  0.78651685  0.84269663  0.79545455]
Training error 0.190796857464

Test acc 0.78469

Previous with AgeFill
---------------------
Cross-validation min 0.753
Cross-validation accuracy 0.816 +/- 0.067
[ 0.78888889  0.8         0.7752809   0.86516854  0.83146067  0.78651685
  0.80898876  0.78651685  0.87640449  0.84090909]
Training error 0.173961840629

Test acc 0.77512

Removing time-based RNG
=======================
Now the results are the same from one run to the next. Also I'm ensuring that the CV splits are shuffled now.

Random Forest, MSS 50, Sex, Class, Fare, AgeFill
------------------------------------------------
Cross-validation min 0.744
Cross-validation accuracy 0.808 +/- 0.081
Training accuracy: 0.841

Random Forest, MSS 50, Sex, Class, Fare
---------------------------------------
Cross-validation min 0.744
Cross-validation accuracy 0.801 +/- 0.079
Training accuracy: 0.837

Random Forest, MSS 80, Sex, Class, Fare, AgeFill
------------------------------------------------
Cross-validation min 0.733
Cross-validation accuracy 0.804 +/- 0.085
Training accuracy: 0.826

Random Forest, MSS 30, Sex, Class, Fare, AgeFill
------------------------------------------------
Cross-validation min 0.778
Cross-validation accuracy 0.822 +/- 0.076
Training accuracy: 0.874

Looks much better.

Random Forest, MSL 5, Sex, Class, Fare, AgeFill
------------------------------------------------
Cross-validation min 0.767
Cross-validation accuracy 0.827 +/- 0.080
Training accuracy: 0.884

Using min samples leaf should allow gender splits at low levels. Overall it's fitting the training data better.

Random Forest, MSL 7, Sex, Class, Fare, AgeFill
------------------------------------------------
Cross-validation min 0.756
Cross-validation accuracy 0.822 +/- 0.080
Training accuracy: 0.871

This is worse all over...

Random Forest, MSL 3, Sex, Class, Fare, AgeFill
------------------------------------------------
Cross-validation min 0.789
Cross-validation accuracy 0.828 +/- 0.068
Training accuracy: 0.899

This seems to do better than MSL 5.

Test acc 0.76077

This is worse than previous experiments with MSS 80.

Random Forest, MSL 3, Sex, Class, FareFill, AgeFill
---------------------------------------------------
Cross-validation min 0.789
Cross-validation accuracy 0.826 +/- 0.063
Training accuracy: 0.900

Random Forest, MSS 80, Sex, Class, FareFill, AgeFill
----------------------------------------------------
Cross-validation min 0.733
Cross-validation accuracy 0.803 +/- 0.083
Training accuracy: 0.825

Test acc 0.78947 (much better!)

Decision Tree
-------------
Cross-validation min 0.722
Cross-validation accuracy 0.789 +/- 0.078
Training accuracy: 0.976

Overfitting by far

Decision Tree, MSS=80
---------------------
Cross-validation min 0.756
Cross-validation accuracy 0.796 +/- 0.065
Training accuracy: 0.832

Test acc: 0.73684 (indeed overfit)

Logistic Regression, Sex, Class, Fare, AgeFill (baseline)
---------------------------------------------------------
Cross-validation min 0.767
Cross-validation accuracy 0.794 +/- 0.047
Training accuracy: 0.800

Logistic Regression, Sex, Class, FareFill, AgeFill
--------------------------------------------------
Cross-validation min 0.767
Cross-validation accuracy 0.796 +/- 0.046
Training accuracy: 0.800

Only slight improvement

Logistic Regression, Sex, Class, FareFill, AgeFill, EmbarkedNums
----------------------------------------------------------------
Cross-validation min 0.722
Cross-validation accuracy 0.783 +/- 0.071
Training accuracy: 0.795

Random Forest, MSS 80, Sex, Class, FareFill, AgeFill, EmbarkedNums
------------------------------------------------------------------
Cross-validation min 0.733
Cross-validation accuracy 0.801 +/- 0.082
Training accuracy: 0.826

Random Forest, MSS 80, Sex, Class, FareFill, AgeFill, EmbarkedNums, DeckNum
---------------------------------------------------------------------------
Cross-validation min 0.744
Cross-validation accuracy 0.816 +/- 0.070
Training accuracy: 0.851

Test acc 0.77512

Random Forest, MSS 80, Sex, Class, FareFill, AgeFill, DeckNum
-------------------------------------------------------------
Cross-validation min 0.744
Cross-validation accuracy 0.807 +/- 0.061
Training accuracy: 0.856

Dropping the embarkation points cause they didn't help on the leaderboard before.
It's fitting CV less well and fitting training data better so I doubt this will be good.

Test acc 0.78469, which is an improvement over the previous but not as good as without DeckNum.

Improving missing value imputation
==================================
Missing values don't need to be imputed purely from the training set - can use the test set
as well because those fields are provided.

Random Forest, MSS 80, Sex, Class, FareFill, AgeFill
----------------------------------------------------
Cross-validation min 0.744
Cross-validation accuracy 0.810 +/- 0.069
Training accuracy: 0.860

Compared to the same table from elsewhere it's an improvement across the board.

Test set acc 0.77990

The test set accuracy is actually worse than just using the training set for this info.

Title counts from all data
--------------------------
Mr              757
Miss            260
Mrs             197
Master           61
Rev               8
Dr                8

Col               4
Major             2
Mlle              2
Ms                2
Mme               1
the Countess      1
Don               1
Lady              1
Sir               1
Jonkheer          1
Dona              1
Capt              1

Imputing age from title
-----------------------
Previously we predicted median(age | Pclass, Sex).
Now it's median(age | Pclass, Title) which backs off to median(age | Title)
because not all Pclass-Title combinations are encountered in the data.

Cross-validation min 0.733
Cross-validation accuracy 0.802 +/- 0.080
Training accuracy: 0.822

Compared to the previous experiment every single number has gone down.

Test set acc 0.77990 (identical)

Decrease min samples split to 10
--------------------------------
Cross-validation min 0.756
Cross-validation accuracy 0.822 +/- 0.071
Training accuracy: 0.910

Increase trees to 500
---------------------
Cross-validation min 0.756
Cross-validation accuracy 0.821 +/- 0.068
Training accuracy: 0.908

Test acc 0.76077

100 trees, MSS 80, max features n-1
-----------------------------------
Cross-validation min 0.733
Cross-validation accuracy 0.812 +/- 0.089
Training accuracy: 0.828

Test acc 0.77512 (actually worse)

100 trees, MSS 80, max features n-2, oob score
-----------------------------------
Cross-validation min 0.744
Cross-validation accuracy 0.820 +/- 0.082
Training accuracy: 0.833

Test acc 0.77033 (worse still!)

Refactoring to clean the unified matrix
=======================================

First test, 100 trees, MSS 80
-----------------------------
Cross-validation min 0.733
Cross-validation accuracy 0.802 +/- 0.080
Training accuracy: 0.822

Test acc 0.77990

Convert to fare per person
--------------------------
Cross-validation min 0.722
Cross-validation accuracy 0.797 +/- 0.077
Training accuracy: 0.817

Test acc 0.75120

Impute fare per person then compute farefill
--------------------------------------------
Cross-validation min 0.733
Cross-validation accuracy 0.802 +/- 0.080
Training accuracy: 0.822

(No difference from the refactor output)
Test acc 0.77990 (same as previous)

Same but 500 trees
------------------
Cross-validation min 0.733
Cross-validation accuracy 0.803 +/- 0.082
Training accuracy: 0.818

Test acc 0.77990

100 trees, n-1 features
-----------------------
Cross-validation min 0.733
Cross-validation accuracy 0.812 +/- 0.080
Training accuracy: 0.829

Test acc 0.77033

100 trees, all features including embarked
------------------------------------------
Cross-validation min 0.733
Cross-validation accuracy 0.804 +/- 0.081
Training accuracy: 0.820

same with title as number
-------------------------
Cross-validation min 0.756
Cross-validation accuracy 0.814 +/- 0.078
Training accuracy: 0.840

same with FamilySize
--------------------
Cross-validation min 0.733
Cross-validation accuracy 0.813 +/- 0.083
Training accuracy: 0.840

same with SibSp, Parch instead of FamilySize
--------------------------------------------
Cross-validation min 0.756
Cross-validation accuracy 0.816 +/- 0.080
Training accuracy: 0.841

Test acc 0.78469

same with MSS 20
----------------
Cross-validation min 0.800
Cross-validation accuracy 0.843 +/- 0.064
Training accuracy: 0.888

Test acc ??? (need to wait a couple hours)

Hyperparameter tuning
=====================
With current feature set this is the best I get
Validation score 0.843
Params {'max_features': None, 'min_samples_split': 20, 'min_samples_leaf': 1, 'criterion': 'gini', 'n_estimators': 100}

If I remove the title and embarked features, I get
Validation score 0.840
Params {'max_features': 2, 'min_samples_split': 20, 'min_samples_leaf': 1, 'criterion': 'gini', 'n_estimators': 100}

I took the gini/entropy and num trees out of the equation

With the deck factorized and cabin number
-----------------------------------------
Deck alone didn't do much of anything.  Cabin number is interesting
but mostly the value is missing.

Cross-validation min 0.800
Cross-validation accuracy 0.846 +/- 0.059
Hyperparameter tuning
Validation score 0.838
Params {'max_features': 11, 'min_samples_split': 10, 'min_samples_leaf': 1}

So it may help but it's tough to say.

With FarePerPersonFill again
----------------------------
Cross-validation min 0.789
Cross-validation accuracy 0.839 +/- 0.064
Hyperparameter tuning
Validation score 0.836
Params {'max_features': 10, 'min_samples_split': 20, 'min_samples_leaf': 4}

Should I use the stratified shuffle split on hyperparam opt?
------------------------------------------------------------
Trying on the same FarePerPersonFill run

Cross-validation min 0.800
Cross-validation accuracy 0.846 +/- 0.059
Hyperparameter tuning
Validation score 0.843
Params {'max_features': 11, 'min_samples_split': 20, 'min_samples_leaf': 1}
...
Training accuracy: 0.889

Doesn't make a difference in the model selected.

Test acc 0.78469 which is good-ish. The graph of cross-validation accuracy vs test acc 
shows that they're correlated after the code refactor so there must've been a horrible bug before.

Retain Fare=0
-------------
I've been assuming that when fare is zero that's an error but it could actually be good info.

Cross-validation min 0.811
Cross-validation accuracy 0.844 +/- 0.057
Hyperparameter tuning
Validation score 0.850
Params {'max_features': 10, 'min_samples_split': 10, 'min_samples_leaf': 2}
...
Training accuracy: 0.907

So it looks like it might be better to leave it in.

Test acc 0.77033

Adding the number of names, switching to grid search
----------------------------------------------------
Cross-validation min 0.789
Cross-validation accuracy 0.831 +/- 0.049

Validation score 0.841
Params {'max_features': 10, 'min_samples_split': 20, 'min_samples_leaf': 2}
Training accuracy: 0.891

It doesn't improve the validation score at all even with hyperparam tuning.

Adding n-4 as max features option
---------------------------------
While doing this I learned that the tests weren't deterministic in part because they
didn't have a random seed set on the forest classifier used in the hyperparameter search.

Validation score 0.838
Params {'max_features': 12, 'min_samples_split': 20, 'min_samples_leaf': 2}

Adding back FarePerPerson
-------------------------
Cross-validation min 0.789
Cross-validation accuracy 0.831 +/- 0.054

Validation score 0.842
Params {'max_features': 11, 'min_samples_split': 20, 'min_samples_leaf': 1}

It seems to help slightly in conjunction with dropping features.

Using regression to fill in age
===============================

Cross-validation min 0.778
Cross-validation accuracy 0.828 +/- 0.059

Validation score 0.836
Params {'max_features': 11, 'min_samples_split': 30, 'min_samples_leaf': 2}

Training accuracy: 0.883

Seems to be much worse but could be generalizing better?

Test acc 0.79426 (best yet of the random forest solutions)

Improving AgeFill regression
----------------------------
The fill_age function can now do cross-validation runs and hyperparameter
tuning. Hyperparameter tuning is fairly important here to prevent overfitting.
Mostly it tunes the number of features and MSS so then I restricted the search to that.

[Age] Cross-validation accuracy 0.412 +/- 0.154
Age hyperparameter tuning
Validation score 0.448, Hyperparams {'max_features': 0.8, 'min_samples_split': 30}

The untuned result didn't set MSS or max features.

Now trying a tuning run that varies the number of trees used (probably won't make much difference).
Age hyperparameter tuning
Validation score 0.448, Hyperparams {'max_features': 0.8, 'min_samples_split': 30, 'n_estimators': 1000}
Validation score 0.448, Hyperparams {'max_features': 0.8, 'min_samples_split': 30, 'n_estimators': 100}

Only slightly better, not worth.

System with a tuned AgeFill regressor
-------------------------------------
Age untuned: 0.412 +/- 0.154
Age tuned: Validation score 0.448, Hyperparams {'max_features': 0.8, 'min_samples_split': 30}

Overall
Validation score 0.837, Hyperparams {'max_features': 11, 'min_samples_split': 20, 'min_samples_leaf': 1}
Validation score 0.837, Hyperparams {'max_features': 10, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.836, Hyperparams {'max_features': 0.5, 'min_samples_split': 30, 'min_samples_leaf': 2}
Training accuracy: 0.899

Test accuracy 0.78469

Switching from StratifiedShuffleSplit to ShuffleSplit
=====================================================
Validation score 0.847, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 2}

I wonder if this will generalize better to testing data without the exact survival ratio.

Test accuracy 0.78469 (no change)

Cabin work
==========
Supposedly odd and even cabin numbers correlated to the side of the boat.

Upon looking into this I realized that setting unknown cabins to U0 actually hurts
me because I'd lump all of those people in with evens.  When splitting it out
by survival clearly odd-side had a higher survival than even side and unknown had
a considerably lower survival rate.

CabinKnown
----------
Dedicated feature for whether the cabin was null or not.

Validation score 0.834, Hyperparams {'max_features': 12, 'min_samples_split': 40, 'min_samples_leaf': 2}

Generally this seems to be worse but it could be because it's already a fixed value in DeckNum and CabinNum

ShipSide, -1, even, odd
-----------------------
Separating unknowns into -1 instead of lumping them with evens.

Validation score 0.834, Hyperparams {'max_features': 12, 'min_samples_split': 20, 'min_samples_leaf': 2}

CabinKnown plus ShipSide
------------------------
Validation score 0.838, Hyperparams {'max_features': 14, 'min_samples_split': 20, 'min_samples_leaf': 2}
Training accuracy: 0.891

DeckNum sort for factorization
------------------------------
Slight improvement on age prediction.
[Age] Cross-validation accuracy 0.404 +/- 0.160
Age hyperparameter tuning
Validation score 0.449, Hyperparams {'max_features': 0.5, 'min_samples_split': 10}

No diff for regular:
Validation score 0.837, Hyperparams {'max_features': 12, 'min_samples_split': 20, 'min_samples_leaf': 2}
Training accuracy: 0.895

Reverting cause it's not 100% clear what it's doing.

FarePerPersonFill_bin
---------------------
Validation score 0.838, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 2}
Training accuracy: 0.893

Slight help

With FareFill_bin (5) and AgeFill_bin (10)
------------------------------------------
Validation score 0.837, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 1}
Training accuracy: 0.895

Dropping CabinNum
-----------------
Age Validation score 0.450, Hyperparams {'max_features': 0.5, 'min_samples_split': 20}
Main Validation score 0.836, Hyperparams {'max_features': 14, 'min_samples_split': 20, 'min_samples_leaf': 1}

With CabinNum back in
---------------------
Data columns: AgeFill, AgeFill_bin, CabinKnown, CabinNum, DeckNum, Embarked_C, Embarked_Q, Embarked_S, FareFill, FareFill_bin, FarePerPersonFill, FarePerPersonFill_bin, NamesNum, Parch, Pclass, SexNum, ShipSide, SibSp, Survived, TitleNum
Validation score 0.837, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 1}
Validation score 0.837, Hyperparams {'max_features': 16, 'min_samples_split': 20, 'min_samples_leaf': 2}

Test acc 0.78469

top was titlenum
 ('SexNum', 0.20474806226900688),
 ('FarePerPersonFill', 0.085249963535382473),
 ('FareFill', 0.084631672540314076),
 ('AgeFill', 0.083165138674732741),
 ('Pclass', 0.072704031013514164),
 ('CabinNum', 0.034360073580464362),
 ('AgeFill_bin', 0.030312032284063851),
 ('NamesNum', 0.029878442153774082),
 ('SibSp', 0.028651725115886616),
 ('FarePerPersonFill_bin', 0.017344837599550408),
 ('DeckNum', 0.017054923541919686),
 ('Embarked_S', 0.012608286868506008),
 ('CabinKnown', 0.010330982834394755),
 ('FareFill_bin', 0.0092749553396701667),
 ('ShipSide', 0.0074096144108947585),
 ('Embarked_C', 0.0065553342407457882),
 ('Parch', 0.0053137484980093977),
 ('Embarked_Q', 0.004201448808146606)]

Adding parameters for missing values
------------------------------------
FareMissing, AgeMissing

Validation score 0.838, Hyperparams {'max_features': 18, 'min_samples_split': 20, 'min_samples_leaf': 1}
Validation score 0.837, Hyperparams {'max_features': 18, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.836, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 1}
Training accuracy: 0.898

[('TitleNum', 0.28852919989873849),
 ('SexNum', 0.19187323727415084),
 ('AgeFill', 0.093645394585899239),
 ('Pclass', 0.09271001168317021),
 ('FarePerPersonFill', 0.090254693459447491),
 ('FareFill', 0.081196138964661199),
 ('CabinNum', 0.03234140492445281),
 ('NamesNum', 0.024027896737747179),
 ('SibSp', 0.023771289663853404),
 ('AgeFill_bin', 0.020340871089120881),
 ('FarePerPersonFill_bin', 0.014314057746875995),
 ('Embarked_S', 0.011540422679047093),
 ('FareFill_bin', 0.0059318034136921658),
 ('Embarked_C', 0.0058050497118136511),
 ('ShipSide', 0.0057531911589752526),
 ('DeckNum', 0.005489411673620259),
 ('Parch', 0.0035903110580161267),
 ('Embarked_Q', 0.0028349128789787893),
 ('AgeMissing', 0.0026291766098801585),
 ('FareMissing', 0.0025317194696012502),
 ('CabinKnown', 0.00088980531825742781)]

Trimming down to almost original features
-----------------------------------------
TitleNum, SexNum, AgeFill, Pclass, FarePerPersonFill

Validation score 0.837, Hyperparams {'max_features': 1, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.836, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.836, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 2}

[('TitleNum', 0.25298124711186981),
 ('SexNum', 0.24877287254740249),
 ('FarePerPersonFill', 0.20878907113430106),
 ('AgeFill', 0.17879464348682816),
 ('Pclass', 0.11066216571959853)]
Training accuracy: 0.874

Test acc 0.76077

Current (just checking)
----------------------
Data columns: AgeFill, AgeFill_bin, AgeMissing, CabinKnown, CabinNum, DeckNum, Embarked_C, Embarked_Q, Embarked_S, FareFill, FareFill_bin, FareMissing, FarePerPersonFill, FarePerPersonFill_bin, NamesNum, Parch, Pclass, SexNum, ShipSide, SibSp, Survived, TitleNum

Validation score 0.838, Hyperparams {'max_features': 18, 'min_samples_split': 20, 'min_samples_leaf': 1}
Validation score 0.837, Hyperparams {'max_features': 18, 'min_samples_split': 20, 'min_samples_leaf': 2}

[('TitleNum', 0.28852919989873849),
 ('SexNum', 0.19187323727415084),
 ('AgeFill', 0.093645394585899239),
 ('Pclass', 0.09271001168317021),
 ('FarePerPersonFill', 0.090254693459447491),
 ('FareFill', 0.081196138964661199),
 ('CabinNum', 0.03234140492445281),
 ('NamesNum', 0.024027896737747179),
 ('SibSp', 0.023771289663853404),
 ('AgeFill_bin', 0.020340871089120881),
 ('FarePerPersonFill_bin', 0.014314057746875995),
 ('Embarked_S', 0.011540422679047093),
 ('FareFill_bin', 0.0059318034136921658),
 ('Embarked_C', 0.0058050497118136511),
 ('ShipSide', 0.0057531911589752526),
 ('DeckNum', 0.005489411673620259),
 ('Parch', 0.0035903110580161267),
 ('Embarked_Q', 0.0028349128789787893),
 ('AgeMissing', 0.0026291766098801585),
 ('FareMissing', 0.0025317194696012502),
 ('CabinKnown', 0.00088980531825742781)]
Training accuracy: 0.898

Dropping Deck G and T
---------------------
Deck G has hundreds of berths but only 5 people in the data so it may contribute to sparseness.
Deck T has only 1 person.

Validation score 0.838, Hyperparams {'max_features': 18, 'min_samples_split': 20, 'min_samples_leaf': 1}
Validation score 0.837, Hyperparams {'max_features': 18, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.836, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 1}

 ('SexNum', 0.19187323727415084),
 ('AgeFill', 0.093645394585899239),
 ('Pclass', 0.09271001168317021),
 ('FarePerPersonFill', 0.090254693459447491),
 ('FareFill', 0.081196138964661199),
 ('CabinNum', 0.03234140492445281),
 ('NamesNum', 0.024027896737747179),
 ('SibSp', 0.023771289663853404),
 ('AgeFill_bin', 0.020340871089120881),
 ('FarePerPersonFill_bin', 0.014314057746875995),
 ('Embarked_S', 0.011540422679047093),
 ('FareFill_bin', 0.0059318034136921658),
 ('Embarked_C', 0.0058050497118136511),
 ('ShipSide', 0.0057531911589752526),
 ('DeckNum', 0.005489411673620259),
 ('Parch', 0.0035903110580161267),
 ('Embarked_Q', 0.0028349128789787893),
 ('AgeMissing', 0.0026291766098801585),
 ('FareMissing', 0.0025317194696012502),
 ('CabinKnown', 0.00088980531825742781)]
Training accuracy: 0.898

Dropping those values makes no difference at all.

Dealing with deck issues by set of booleans
-------------------------------------------
I wanted to replace but the deck is used by AgeFill and I got lazy.

Validation score 0.837, Hyperparams {'max_features': 26, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.836, Hyperparams {'max_features': 29, 'min_samples_split': 20, 'min_samples_leaf': 2}

[('TitleNum', 0.30822693057044803),
 ('SexNum', 0.18220918236472353),
 ('Pclass', 0.093103457852434665),
 ('AgeFill', 0.091649154013464018),
 ('FarePerPersonFill', 0.090941310630972175),
 ('FareFill', 0.075879814613958013),
 ('CabinNum', 0.02778511920250764),
 ('SibSp', 0.025134475570430647),
 ('NamesNum', 0.022748698497516037),
 ('AgeFill_bin', 0.017904817521553734),
 ('FarePerPersonFill_bin', 0.017032018299677088),
 ('Embarked_S', 0.011362753118255804),
 ('FareFill_bin', 0.0055175255158318101),
 ('Embarked_C', 0.005103451010616664),
 ('ShipSide', 0.0036378691338123902),
 ('DeckNum', 0.0033136668919036354),
 ('Deck_D', 0.0028604568937684927),
 ('Parch', 0.0025190299186638766),
 ('Deck_U', 0.002444551103227824),
 ('Embarked_Q', 0.0022301353748477594),
 ('AgeMissing', 0.002135871840070165),
 ('FareMissing', 0.0018316174903305546),
 ('Deck_C', 0.0015485043212398186),
 ('Deck_E', 0.0014415109092233275),
 ('CabinKnown', 0.00054925899698078868),
 ('Deck_A', 0.00048507770766712172),
 ('Deck_B', 0.00022540211393827823),
 ('Deck_G', 0.00017833852193634194),
 ('Deck_F', 0.0),
 ('Deck_T', 0.0)]
 
Trying again without DeckNum
----------------------------
I want to do this so I can better see the importance of individual decks.

Validation score 0.836, Hyperparams {'max_features': 27, 'min_samples_split': 20, 'min_samples_leaf': 1}
Validation score 0.836, Hyperparams {'max_features': 27, 'min_samples_split': 30, 'min_samples_leaf': 2}

[('TitleNum', 0.32177931289593886),
 ('SexNum', 0.16089143603894399),
 ('Pclass', 0.096275238582898409),
 ('AgeFill', 0.096269259356081244),
 ('FarePerPersonFill', 0.087917490867944106),
 ('FareFill', 0.083247397016026861),
 ('CabinNum', 0.026019270243005762),
 ('SibSp', 0.024222985528401443),
 ('NamesNum', 0.023463461766993151),
 ('FarePerPersonFill_bin', 0.015849071812469629),
 ('AgeFill_bin', 0.014855945005216378),
 ('Embarked_S', 0.0099190329769055612),
 ('Embarked_C', 0.0052526097455421264),
 ('ShipSide', 0.0045194945309711201),
 ('Deck_D', 0.0044791874809121781),
 ('FareFill_bin', 0.0040302803876842439),
 ('Deck_E', 0.0033832932491146672),
 ('Embarked_Q', 0.0031533989461049578),
 ('Parch', 0.00298673344879344),
 ('FareMissing', 0.002983772932617001),
 ('AgeMissing', 0.0023967962533390278),
 ('Deck_C', 0.0021030161898082066),
 ('Deck_U', 0.0018619656985326006),
 ('Deck_B', 0.00074418933615437053),
 ('Deck_G', 0.00048969928767380623),
 ('Deck_A', 0.00044447420534475009),
 ('Deck_F', 0.00027626240226402015),
 ('CabinKnown', 0.00018492381431812045),
 ('Deck_T', 0.0)]
Training accuracy: 0.901

It's interesting that decks D and E were so much more useful than the others. Also,
the training accuracy shows that it's fitting the training data better although the
cross-validation accuracy is showing that it's not having much effect.

Regression to fill FarePerPersonFill
====================================
Features used in pivot table (Pclass, Embarked): 0.385
Lots more features: 0.456

There are some interesting correlations revealed:
[('Pclass', 0.34695167299090024),
 ('CabinNum', 0.26798530233475215),
 ('DeckNum', 0.11371077178301751),
 ('TitleNum', 0.086196621572053087),
 ('SibSp', 0.047973308300285719),
 ('SexNum', 0.041124393034330679),
 ('Embarked_C', 0.036858146025249623),
 ('Parch', 0.03644699028888914),
 ('Embarked_S', 0.022178504246547891),
 ('Embarked_Q', 0.00057428942397400474)]

Apparently front vs back of the ship correlated with price in addition to the deck (beyond what's already covered in passenger class).

Validation score 0.839, Hyperparams {'max_features': 28, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.839, Hyperparams {'max_features': 27, 'min_samples_split': 30, 'min_samples_leaf': 2}
Validation score 0.839, Hyperparams {'max_features': 25, 'min_samples_split': 20, 'min_samples_leaf': 2}

[('TitleNum', 0.31738559605174105),
 ('SexNum', 0.17102081976362032),
 ('FarePerPersonFill', 0.10094473181227791),
 ('Pclass', 0.099528346628104933),
 ('AgeFill', 0.09261663119909104),
 ('FareFill', 0.084198471164106892),
 ('CabinNum', 0.022529207557394817),
 ('NamesNum', 0.02145634275055321),
 ('SibSp', 0.020100640591673713),
 ('AgeFill_bin', 0.017952978393556234),
 ('Embarked_S', 0.010962305571015494),
 ('FarePerPersonFill_bin', 0.008890742589695081),
 ('FareFill_bin', 0.0072273450792268842),
 ('Embarked_C', 0.0043780528795076771),
 ('Deck_D', 0.0040894844112964717),
 ('Embarked_Q', 0.003233474793259924),
 ('Parch', 0.0032075658666518496),
 ('AgeMissing', 0.0025929292960524364),
 ('ShipSide', 0.0023188569078955988),
 ('Deck_C', 0.0021804366670038137),
 ('Deck_U', 0.001158471723809025),
 ('Deck_E', 0.00075373602376910904),
 ('CabinKnown', 0.00054736036109204406),
 ('Deck_G', 0.00023816705089190152),
 ('FareMissing', 0.00023683748580053863),
 ('Deck_A', 0.00013572246034450366),
 ('Deck_B', 0.00011474492056758604),
 ('Deck_F', 0.0),
 ('Deck_T', 0.0)]

The validation score improved and also the FarePerPersonFill is used more. Interestingly FareFill isn't used more.

Reverting Deck indicators (back to just DeckNum)
------------------------------------------------
I like the idea of deck indicator vars but it didn't lead to results and it clogs stuff up.

Validation score 0.840, Hyperparams {'max_features': 18, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.839, Hyperparams {'max_features': 19, 'min_samples_split': 20, 'min_samples_leaf': 2}

[('TitleNum', 0.2704497753437593),
 ('SexNum', 0.21494892487836245),
 ('FarePerPersonFill', 0.098169964567186158),
 ('AgeFill', 0.095009010996103274),
 ('Pclass', 0.093869028527061124),
 ('FareFill', 0.087189865304258748),
 ('SibSp', 0.022550756636560556),
 ('CabinNum', 0.022184358609879867),
 ('AgeFill_bin', 0.019728159086831117),
 ('NamesNum', 0.018985729682183601),
 ('Embarked_S', 0.012334476593503684),
 ('FareFill_bin', 0.0097769395320988356),
 ('FarePerPersonFill_bin', 0.0094821176558827335),
 ('DeckNum', 0.0055975987013279459),
 ('Embarked_C', 0.0049500566466286781),
 ('ShipSide', 0.0041434613875442651),
 ('AgeMissing', 0.0034376467037596976),
 ('Embarked_Q', 0.0028087237840156588),
 ('Parch', 0.0022093490952245268),
 ('CabinKnown', 0.0020095802444111861),
 ('FareMissing', 0.00016447602341657493)]
 Training accuracy: 0.888

Test acc: 0.77990

Ooops, handling zeros
---------------------
I had accidentally retained the zeros as non-null in the regression model. Fixing that:

[FarePerPerson] Cross-validation accuracy 0.507 +/- 0.437
FarePerPerson hyperparameter tuning
    [('Pclass', 0.36186416722949422),
     ('CabinNum', 0.27761769409379711),
     ('DeckNum', 0.09926518082919715),
     ('TitleNum', 0.090662902102306667),
     ('SibSp', 0.04807269321235081),
     ('SexNum', 0.037272759135274898),
     ('Parch', 0.03483738335952237),
     ('Embarked_C', 0.031812421835007217),
     ('Embarked_S', 0.017569707085801897),
     ('Embarked_Q', 0.0010250911172476314)]
Validation score 0.548, Hyperparams {'max_features': 0.5, 'min_samples_split': 10}

Much higher correlation with actual fare values!

Hyperparameter tuning
Validation score 0.839, Hyperparams {'max_features': 18, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.838, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.837, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 1}
     ('SexNum', 0.21325674711413969),
     ('Pclass', 0.099657361566270378),
     ('AgeFill', 0.091831372280595042),
     ('FarePerPersonFill', 0.087506272959800971),
     ('FareFill', 0.082018575089548018),
     ('CabinNum', 0.030419167643351799),
     ('SibSp', 0.022236683678012945),
     ('NamesNum', 0.02097965199984804),
     ('AgeFill_bin', 0.01879022109445562),
     ('FarePerPersonFill_bin', 0.015797173485369166),
     ('Embarked_S', 0.011841519290270964),
     ('DeckNum', 0.0058249345007911424),
     ('Embarked_C', 0.0056962514609446933),
     ('ShipSide', 0.0042970793812439814),
     ('FareFill_bin', 0.0038811894875893684),
     ('Parch', 0.0031146385176945434),
     ('AgeMissing', 0.0028053133306200728),
     ('Embarked_Q', 0.0026898467593811683),
     ('CabinKnown', 0.0019872379429453878),
     ('FareMissing', 0.0012662542587692548)]
Training accuracy: 0.891

So it drops back down to match FareFill now.

Test acc: 0.78469

Aside: Feature importance for age regression
--------------------------------------------
    [('TitleNum', 0.34160655195997441),
     ('Pclass', 0.24647747370639775),
     ('Parch', 0.20992342073071477),
     ('CabinNum', 0.063020153727720463),
     ('SibSp', 0.039638977781503291),
     ('NamesNum', 0.035176506626663065),
     ('SexNum', 0.016310776931930081),
     ('DeckNum', 0.014620124362137801),
     ('Embarked_Q', 0.012946197989578294),
     ('Embarked_C', 0.012108619381220806),
     ('Embarked_S', 0.0081711968021592977)]
Validation score 0.448, Hyperparams {'max_features': 0.8, 'min_samples_split': 30}

Learning curves
===============
I generated some learning curves with MSS off and saw that
without regularization it's overfitting insanely hard.

Ticket groups
=============
Families tend to live or die together so we can factorize the tickets that are shared. If a child
died in a group then probably everyone did.

Validation score 0.846, Hyperparams {'max_features': 15, 'min_samples_split': 20, 'min_samples_leaf': 1}
Validation score 0.844, Hyperparams {'max_features': 16, 'min_samples_split': 20, 'min_samples_leaf': 1}

    [('TitleNum', 0.30143857435832011),
     ('SexNum', 0.17122365473270931),
     ('FarePerPersonFill', 0.095862990641113754),
     ('AgeFill', 0.092603327360187393),
     ('Pclass', 0.089188451570627811),
     ('FareFill', 0.063736526495442036),
     ('GroupSize', 0.051661722811222298),
     ('TicketNum', 0.033849472504318509),
     ('CabinNum', 0.03076509662278883),
     ('NamesNum', 0.023988562335613896),
     ('Embarked_S', 0.012240958517187664),
     ('SibSp', 0.012181807051220255),
     ('DeckNum', 0.0060646398617351729),
     ('ShipSide', 0.0049639504662842681),
     ('Embarked_Q', 0.0037366573930965332),
     ('Embarked_C', 0.0034433562742133611),
     ('Parch', 0.0030502510039187446)]
Training accuracy: 0.905

The actual ticket number seems to be useful as is the GroupSize even though SibSp and Parch aren't as good.

Test acc: 0.75598

Yeah I thought it might be too good to be true...

Parsing the ticket
==================
The ticket has an optional letter/symb prefix then numeric part

Baseline
--------
Validation score 0.840, Hyperparams {'max_features': 11, 'min_samples_split': 20, 'min_samples_leaf': 2}

    [('TitleNum', 0.30529969184249733),
     ('SexNum', 0.18868649109296753),
     ('FarePerPersonFill', 0.11402348615872868),
     ('AgeFill', 0.10671057462663373),
     ('FareFill', 0.090710441942740333),
     ('Pclass', 0.085341297406080568),
     ('SibSp', 0.024453723885153558),
     ('CabinNum', 0.022979405869169245),
     ('NamesNum', 0.021157173671131693),
     ('DeckNum', 0.012171138500984828),
     ('Embarked_S', 0.010381711752138898),
     ('ShipSide', 0.0068362939980875295),
     ('Embarked_C', 0.0046964469208884479),
     ('Embarked_Q', 0.0033380285789772098),
     ('Parch', 0.0032140937538205228)]

With ticket alpha and num features
----------------------------------
I reduced the ticket alpha part from 50 values to 39 by stripping periods and uppercasing.

Validation score 0.851, Hyperparams {'max_features': 14, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.847, Hyperparams {'max_features': 16, 'min_samples_split': 20, 'min_samples_leaf': 2}

    [('TitleNum', 0.31149452344675771),
     ('SexNum', 0.16916081357830515),
     ('TicketNumPart', 0.093895773919168771),
     ('FarePerPersonFill', 0.08966329162337526),
     ('Pclass', 0.082940894035589616),
     ('AgeFill', 0.079221371560151563),
     ('FareFill', 0.068924862800870554),
     ('CabinNum', 0.02579922937381203),
     ('SibSp', 0.021754070832896738),
     ('NamesNum', 0.01855313951261919),
     ('TicketAlphaPart', 0.012000351894506741),
     ('Embarked_S', 0.010339192432052651),
     ('ShipSide', 0.0049664159821416354),
     ('DeckNum', 0.0043848649894691893),
     ('Parch', 0.0028730860721950763),
     ('Embarked_C', 0.0022819641907187173),
     ('Embarked_Q', 0.0017461537553692763)]
 
I'd guess that ticketnumpart is grossly overfitting but I'll submit it anyway.

Test acc 0.78947

I'm amazed that it generalized at all.

Forcing generalization of ticket num part
-----------------------------------------
If I take the ticket number mod 100 it should generalize a little better... maybe.

Validation score 0.851, Hyperparams {'max_features': 14, 'min_samples_split': 30, 'min_samples_leaf': 1}
Validation score 0.851, Hyperparams {'max_features': 14, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.849, Hyperparams {'max_features': None, 'min_samples_split': 20, 'min_samples_leaf': 1}

Although the model uses both features I want to try and remove the original ticket num.

Removing non-mod ticket num
---------------------------
    Validation score 0.842, Hyperparams {'max_features': 16, 'min_samples_split': 30, 'min_samples_leaf': 2}
    Validation score 0.842, Hyperparams {'max_features': 13, 'min_samples_split': 30, 'min_samples_leaf': 2}
    Validation score 0.841, Hyperparams {'max_features': 16, 'min_samples_split': 30, 'min_samples_leaf': 1}

So it doesn't fit the data quite as well.

    [('TitleNum', 0.33680435214742116),
     ('SexNum', 0.17878643142588529),
     ('Pclass', 0.09865900338016742),
     ('FarePerPersonFill', 0.086162099127473704),
     ('AgeFill', 0.073070348985535119),
     ('FareFill', 0.071241969612890421),
     ('TicketNumPartMod', 0.066913353048752747),
     ('CabinNum', 0.022959048469508934),
     ('NamesNum', 0.015441732433760182),
     ('SibSp', 0.014890939901764296),
     ('TicketAlphaPart', 0.010835117523263662),
     ('Embarked_S', 0.0095135024172817613),
     ('ShipSide', 0.0048542651772930743),
     ('DeckNum', 0.0031648618852887275),
     ('Embarked_C', 0.0031167978735730461),
     ('Parch', 0.0020936108364381649),
     ('Embarked_Q', 0.0014925657537023173)]
 
Data columns: AgeFill, CabinNum, DeckNum, Embarked_C, Embarked_Q, Embarked_S, FareFill, FarePerPersonFill, NamesNum, Parch, Pclass, SexNum, ShipSide, SibSp, Survived, TicketAlphaPart, TicketNumPartMod, TitleNum

Test acc: 0.77990 (worse)

Ooops I meant divided by 100...
-------------------------------
I was lazy and did TicketNum -= TicketNum % 100

Validation score 0.847, Hyperparams {'max_features': 13, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.844, Hyperparams {'max_features': None, 'min_samples_split': 20, 'min_samples_leaf': 2}

    [('TitleNum', 0.29909996288669771),
     ('SexNum', 0.18291339565817785),
     ('Pclass', 0.087649451622814054),
     ('FarePerPersonFill', 0.087500348811328704),
     ('AgeFill', 0.084742549761543759),
     ('FareFill', 0.074273168932331157),
     ('TicketNumPart', 0.068107632716968208),
     ('CabinNum', 0.025492227589917742),
     ('SibSp', 0.022122885182704347),
     ('NamesNum', 0.019196739653118088),
     ('TicketAlphaPart', 0.013596104899811621),
     ('Embarked_S', 0.0125707093737861),
     ('ShipSide', 0.0086117620848640518),
     ('DeckNum', 0.006471627097016589),
     ('Embarked_C', 0.0033417495549456578),
     ('Parch', 0.0021705239155921912),
     ('Embarked_Q', 0.0021391602583820551)]
Training accuracy: 0.897

The accuracy is higher and the ticket num part gets higher weight.

Test acc: 0.77990

Reassessing tuning params
-------------------------
It's picking min leaf 2 much more now so I'll add 3.  I tried then saw 3 was max so added 4 too. I removed
some of the min sample split to compensate.

Validation score 0.851, Hyperparams {'max_features': 14, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.850, Hyperparams {'max_features': 15, 'min_samples_split': 20, 'min_samples_leaf': 3}
Validation score 0.849, Hyperparams {'max_features': 13, 'min_samples_split': 20, 'min_samples_leaf': 4}

So it's not doing too badly.

Converting title to indicator features
--------------------------------------
Validation score 0.846, Hyperparams {'max_features': 21, 'min_samples_split': 20, 'min_samples_leaf': 3}
Validation score 0.843, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 3}
Validation score 0.842, Hyperparams {'max_features': 22, 'min_samples_split': 20, 'min_samples_leaf': 2}

This performs worse!

[('Title_Mr', 0.2926882391355165),
 ('SexNum', 0.17617610645674625),
 ('TicketNumPart', 0.091324305036141029),
 ('FarePerPersonFill', 0.08529162492463882),
 ('Pclass', 0.081457741252890473),
 ('AgeFill', 0.076082032737402758),
 ('FareFill', 0.067149635184338807),
 ('SibSp', 0.026044297059395975),
 ('CabinNum', 0.025063450368813502),
 ('NamesNum', 0.016238380876034991),
 ('TicketAlphaPart', 0.010786771144392485),
 ('Embarked_S', 0.010255557247479148),
 ('Title_Master', 0.0092406575526356831),
 ('Title_Rev', 0.008438795449064649),
 ('DeckNum', 0.0065841602314271616),
 ('ShipSide', 0.0032754175035566773),
 ('Title_Miss', 0.0030015805918950757),
 ('Parch', 0.0024334188154110582),
 ('Title_Dr', 0.0023440052560170874),
 ('Embarked_C', 0.0023005737209823539),
 ('Embarked_Q', 0.001990966691117224),
 ('Title_Mrs', 0.00074474926168588309),
 ('Title_Military', 0.00063762714379569365),
 ('Title_Sir', 0.00044990635862083428),
 ('Title_Lady', 0.0)]

I kind of prefer this though because the trees can ignore Sir and Lady.

Retuning to FarePerPersonFill
-----------------------------
- Switching from factorized Title to indicator Title, helps only a tiny bit but requires min_samples_leaf tuning.
- Adding TicketAlphaPart helpful 0.549 -> 0.569
- Don't really want to add TicketNumPart

Validation score 0.848, Hyperparams {'max_features': 21, 'min_samples_split': 20, 'min_samples_leaf': 2}
Validation score 0.844, Hyperparams {'max_features': 21, 'min_samples_split': 20, 'min_samples_leaf': 3}
Validation score 0.843, Hyperparams {'max_features': 21, 'min_samples_split': 20, 'min_samples_leaf': 4}

    [('Title_Mr', 0.28539688309792838),
     ('SexNum', 0.17132954236406078),
     ('FarePerPersonFill', 0.089302314887669793),
     ('TicketNumPart', 0.08729091763939141),
     ('Pclass', 0.085949277291770154),
     ('AgeFill', 0.079187293211847309),
     ('FareFill', 0.072143352664065533),
     ('CabinNum', 0.023253985102203726),
     ('SibSp', 0.022705671983139636),
     ('NamesNum', 0.016062294829703443),
     ('TicketAlphaPart', 0.0128317649987248),
     ('Title_Rev', 0.01101211614230586),
     ('Embarked_S', 0.0094532223635907988),
     ('DeckNum', 0.0076336756373551898),
     ('Title_Master', 0.004842667080675437),
     ('ShipSide', 0.0044292151265404971),
     ('Embarked_C', 0.0036173289113890809),
     ('Title_Dr', 0.003237039668546703),
     ('Title_Miss', 0.0027770685957148439),
     ('Parch', 0.002480260040300677),
     ('Embarked_Q', 0.0021671179510456173),
     ('Title_Military', 0.0014948636497379075),
     ('Title_Sir', 0.0011544576098024477),
     ('Title_Mrs', 0.00024766915248994754),
     ('Title_Lady', 0.0)]

Looks like a slight improvement though usually FarePPF and FareF are more closely used. Actually not in the last test.

Note
----
Figured out that you can get std dev of cross validation scores so now I'm showing those.
That could help a bit in model selection.

Submitting
----------
The main changes since last time is a slight improvement on FarePPF and additional tuning on AgeF.

Validation score 0.848 +/- 0.036, Hyperparams {'max_features': 21, 'min_samples_split': 20, 'min_samples_leaf': 2}
Data columns: AgeFill, CabinNum, DeckNum, Embarked_C, Embarked_Q, Embarked_S, FareFill, FarePerPersonFill, NamesNum, Parch, Pclass, SexNum, ShipSide, SibSp, Survived, TicketAlphaPart, TicketNumPart, Title_Dr, Title_Lady, Title_Master, Title_Military, Title_Miss, Title_Mr, Title_Mrs, Title_Rev, Title_Sir
Training accuracy: 0.901

Test acc: 0.78469

Fixing CabinNumber when there are multiple cabin numbers
========================================================

Validation score 0.848 +/- 0.036, Hyperparams {'max_features': 21, 'min_samples_split': 20, 'min_samples_leaf': 2}
Feature importances
	Title_Mr            : 0.285396883098
	SexNum              : 0.171329542364
	FarePerPersonFill   : 0.0893023148877
	TicketNumPart       : 0.0872909176394
	Pclass              : 0.0859492772918
	AgeFill             : 0.0791872932118
	FareFill            : 0.0721433526641
	CabinNum            : 0.0232539851022
	SibSp               : 0.0227056719831
	NamesNum            : 0.0160622948297
	TicketAlphaPart     : 0.0128317649987
	Title_Rev           : 0.0110121161423
	Embarked_S          : 0.00945322236359
	DeckNum             : 0.00763367563736
	Title_Master        : 0.00484266708068
	ShipSide            : 0.00442921512654
	Embarked_C          : 0.00361732891139
	Title_Dr            : 0.00323703966855
	Title_Miss          : 0.00277706859571
	Parch               : 0.0024802600403
	Embarked_Q          : 0.00216711795105
	Title_Military      : 0.00149486364974
	Title_Sir           : 0.0011544576098
	Title_Mrs           : 0.00024766915249
	Title_Lady          : 0.0
Training accuracy: 0.901

Fare prediction uses CabinNum extensively:
Feature importances
	Pclass              : 0.321198066893
	CabinNum            : 0.229890373491
	Title_Miss          : 0.107535277364
	TicketAlphaPart     : 0.101437730328
	DeckNum             : 0.0981445914002
	SibSp               : 0.0375326034474
	Parch               : 0.0327145249364
	Embarked_C          : 0.0276498406751
	SexNum              : 0.0167701007566
	Embarked_S          : 0.0109487386891
	Title_Mr            : 0.00944591237633
	Title_Mrs           : 0.00576419450351
	Title_Lady          : 0.000355773173703
	Embarked_Q          : 0.000256570265146
	Title_Military      : 0.000120282180539
	Title_Master        : 0.000111374269969
	Title_Sir           : 8.84737749018e-05
	Title_Dr            : 2.4144480537e-05
	Title_Rev           : 1.14269939885e-05
Validation score 0.569 +/- 0.182, Hyperparams {'max_features': 0.5, 'min_samples_split': 10, 'min_samples_leaf': 2}

Fixing CabinNum
===============
The code took the average of cabin numbers when multiple were listed. This
could lead to setting the side of the ship incorrectly because the average of 10 and 12 is 11.
So now I take the median without averaging for even arrays.

This improves fare prediction slightly:
Feature importances
    Pclass              : 0.314551785943
    CabinNum            : 0.255943410809
    Title_Miss          : 0.104307257371
    TicketAlphaPart     : 0.10288695025
    DeckNum             : 0.0906550842839
    Parch               : 0.0320772931554
    SibSp               : 0.0276003906018
    Embarked_C          : 0.0244670207273
    SexNum              : 0.0180771169647
    Title_Mrs           : 0.00933224839913
    Embarked_S          : 0.00923415337775
    Title_Mr            : 0.00657203890719
    Title_Lady          : 0.00167906738524
    Title_Military      : 0.00142713741205
    Embarked_Q          : 0.000477293097015
    Title_Sir           : 0.000286932320547
    Title_Master        : 0.000221291278761
    Title_Dr            : 0.000183180696119
    Title_Rev           : 2.03470187335e-05
Validation score 0.570 +/- 0.181, Hyperparams {'max_features': 0.5, 'min_samples_split': 10, 'min_samples_leaf': 1}

The overall score is degraded slightly:
Validation score 0.846 +/- 0.040, Hyperparams {'max_features': 21, 'min_samples_split': 20, 'min_samples_leaf': 3}
Feature importances
	Title_Mr            : 0.290910389366
	SexNum              : 0.175597887013
	Pclass              : 0.0866548453727
	FarePerPersonFill   : 0.08619839242
	TicketNumPart       : 0.0856178581148
	AgeFill             : 0.0772175299592
	FareFill            : 0.0743992115092
	CabinNum            : 0.0226584688498
	SibSp               : 0.0223045867299
	NamesNum            : 0.0144738726636
	Title_Rev           : 0.0108013782504
	TicketAlphaPart     : 0.00987318928824
	Embarked_S          : 0.00960854584672
	DeckNum             : 0.00765880937505
	Title_Master        : 0.00518978466608
	ShipSide            : 0.00517922863587
	Title_Miss          : 0.00296805328117
	Embarked_C          : 0.00279450394293
	Embarked_Q          : 0.00274320043861
	Parch               : 0.00268191756272
	Title_Dr            : 0.00249530660084
	Title_Military      : 0.00110045250253
	Title_Sir           : 0.000595931903608
	Title_Mrs           : 0.000276655706581
	Title_Lady          : 0.0
Training accuracy: 0.895

It does result in a higher weight on the ship side.

Test acc 0.78469 (no change)

Experiments with fixing the overfitting
=======================================

Test 1: Retain only the original 3 features
-------------------------------------------
Reproducing the Random Forest with SexNum, Pclass, and FareFillBin.

This isn't entirely the same as the original because FareFill has a lot of work going into it.
The baseline graph is horribly overfit to training data and test acc goes to about 0.825.

The graph with just 3 features is interesting. The training and CV error are very close.
The training stddev is very low and the testing stddev is quite high.

Validation score 0.796 +/- 0.040, Hyperparams {'max_features': None, 'min_samples_split': 20, 'min_samples_leaf': 1}
Feature importances
	SexNum              : 0.694052351179
	Pclass              : 0.226150542923
	FareFillBin         : 0.079797105898

Test acc 0.77990

Test 1.1: Without complex FareFill logic
----------------------------------------
Computing directly from Fare.

Validation score 0.796 +/- 0.040, Hyperparams {'max_features': None, 'min_samples_split': 20, 'min_samples_leaf': 1}
Feature importances
	SexNum              : 0.687217435508
	Pclass              : 0.220504198241
	FareFillBin         : 0.0922783662512

Basically no difference.

Test acc 0.77990

Test 1.2: 2 MSS
---------------
Added MSS 2, 5 and MSS 2 was picked though with the same scores:
Validation score 0.796 +/- 0.040, Hyperparams {'max_features': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
Feature importances
	SexNum              : 0.687043318632
	Pclass              : 0.220482908241
	FareFillBin         : 0.0924737731275

Test acc 0.77990

Somehow or other the original outperforms...

Test 2: Adding AgeFill
----------------------
Simply adding age adds a TON of variance to the data and it's easy to see in the graph.

Validation score 0.828 +/- 0.030, Hyperparams {'max_features': None, 'min_samples_split': 5, 'min_samples_leaf': 2}
Feature importances
	SexNum              : 0.403798111361
	AgeFill             : 0.352665354345
	Pclass              : 0.157650248163
	FareFillBin         : 0.0858862861305

But note that the validation score stddev goes down.

Test acc 0.72727 (!)

Test 2.1: AgeFillBin
--------------------
Grouping Age in 5-year bins to reduce variance.

With max 12 bins
Validation score 0.824 +/- 0.031, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 2, 'min_samples_leaf': 2}
Feature importances
	SexNum              : 0.465432373931
	AgeFillBin          : 0.230180773815
	Pclass              : 0.182419040448
	FareFillBin         : 0.121967811805
Training accuracy: 0.855

With max 10 bins
Validation score 0.829 +/- 0.027, Hyperparams {'max_features': 1, 'min_samples_split': 5, 'min_samples_leaf': 2}
Feature importances
	SexNum              : 0.476482639271
	AgeFillBin          : 0.208939002292
	Pclass              : 0.180816195067
	FareFillBin         : 0.133762163369
Training accuracy: 0.855

With max 8 bins
The graph looks quite a bit better but what about validation scores?
Validation score 0.824 +/- 0.032, Hyperparams {'max_features': None, 'min_samples_split': 2, 'min_samples_leaf': 2}
Feature importances
	SexNum              : 0.521934800876
	Pclass              : 0.209286875416
	AgeFillBin          : 0.163192700171
	FareFillBin         : 0.105585623536

It's getting worse but in some sense it's just because it's depending less on AgeFillBin.

Test acc: 0.77512 (still worse than excluding age entirely)

Test 3: Common title indicators
-------------------------------
Dropping age out entirely what about just title indicators? I dropped the Dr/Lady/Military/Rev/Sir ones cause they're sparse.

Validation score 0.821 +/- 0.043, Hyperparams {'max_features': 6, 'min_samples_split': 40, 'min_samples_leaf': 1}

Feature importances
	Title_Mr            : 0.385500747826
	SexNum              : 0.303190572145
	Pclass              : 0.203785047022
	FareFillBin         : 0.0832838815614
	Title_Master        : 0.0206520102334
	Title_Miss          : 0.00189242127227
	Title_Mrs           : 0.00169531993944

The validation curve looks excellent for this one.

Test acc: 0.78947 (pretty huge improvement)

Test 4: Ship side (very sparse)
-------------------------------
The side of the ship is quite sparse but may be enough to help when available.

Validation score 0.820 +/- 0.041, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 20, 'min_samples_leaf': 3}
Feature importances
	Title_Mr            : 0.316336458337
	SexNum              : 0.205367111242
	Pclass              : 0.157266538369
	FareFillBin         : 0.0885640919222
	ShipSide            : 0.0812499156447
	Title_Miss          : 0.0767618925466
	Title_Mrs           : 0.0556583049664
	Title_Master        : 0.0187956869722

This isn't helping at all.

Test 5: Ticket alpha
--------------------
Validation score 0.826 +/- 0.032, Hyperparams {'max_features': 5, 'min_samples_split': 20, 'min_samples_leaf': 1}
Feature importances
	Title_Mr            : 0.354884009718
	SexNum              : 0.23474764839
	Pclass              : 0.187694870558
	FareFillBin         : 0.0894849178436
	TicketAlphaPart     : 0.0607229139361
	Title_Master        : 0.0309170293874
	Title_Miss          : 0.0228660116143
	Title_Mrs           : 0.0186825985526

Learning curve shows the ticket alpha part overfitting a ton but maybe with the added regularization it'll
bump up the testing acc.

Test acc: 0.79426 (best RF one yet)

Test 6: Age again
-----------------
I'm going to try AgeFillBin using qcut this time but in fewer categories, only 5.

Validation score 0.831 +/- 0.027, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 20, 'min_samples_leaf': 1}
Feature importances
	Title_Mr            : 0.247315418224
	SexNum              : 0.237781647949
	Pclass              : 0.159503806349
	FareFillBin         : 0.101594567554
	AgeFillBin          : 0.0700875832397
	TicketAlphaPart     : 0.0621416535338
	Title_Mrs           : 0.0555418686883
	Title_Miss          : 0.0470671544816
	Title_Master        : 0.0189662999801
(Graph is overfitting more but regularization maaaybe will solve it)

Test acc: 0.77990 (drat)

Age in 5 bins as indicator variables
------------------------------------
There's a small chance that converting age to indicator variables will reduce overfitting.

Validation score 0.830 +/- 0.023, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 1}
Feature importances
	Title_Mr            : 0.29922874867
	SexNum              : 0.233257421615
	Pclass              : 0.168463912585
	FareFillBin         : 0.0911970617017
	TicketAlphaPart     : 0.0563236567776
	AgeFillBin          : 0.0294281162379
	Title_Mrs           : 0.0273693625674
	Title_Miss          : 0.0239932312363
	Title_Master        : 0.0200261005118
	AgeFillBin_3        : 0.012748429256
	AgeFillBin_1        : 0.0119954401295
	AgeFillBin_4        : 0.0111022245335
	AgeFillBin_0        : 0.00849118380908
	AgeFillBin_2        : 0.00637511036971

Eh doesn't look like it's doing much to be honest except overfitting when I look at the graph.

FareFill from RF regression
---------------------------
FareFill R^2 
Validation score 0.722 +/- 0.114, Hyperparams {'max_features': 0.8, 'min_samples_split': 10, 'min_samples_leaf': 1}

The R^2 is much higher than trying to fill in fare per person.

The learning curve looks much much smoother than converting backwards from FareFillPerPerson.

Validation score 0.824 +/- 0.034, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 1}
Feature importances
	Title_Mr            : 0.295778859123
	SexNum              : 0.273728365863
	Pclass              : 0.185046841138
	FareFillBin         : 0.0905311560161
	TicketAlphaPart     : 0.0608526350419
	Title_Mrs           : 0.0336063587979
	Title_Miss          : 0.0307904675266
	Title_Master        : 0.0296653164937

Compared to ticket alpha this looks worse but I have a good feeling about it.

Test acc: 0.79426 (same as before)

Adding TicketSize to FareFill
-----------------------------
Fare hyperparameter tuning
Feature importances
	TicketSize          : 0.422876820229
	Pclass              : 0.383775854628
	CabinNum            : 0.0704656654573
...
Validation score 0.885 +/- 0.059, Hyperparams {'max_features': 0.8, 'min_samples_split': 10, 'min_samples_leaf': 1}

Seems like much much better correlation.

Validation score 0.826 +/- 0.034, Hyperparams {'max_features': 0.5, 'min_samples_split': 30, 'min_samples_leaf': 1}
Feature importances
	SexNum              : 0.282727497802
	Title_Mr            : 0.252210374982
	Pclass              : 0.146390542198
	TicketSize          : 0.107025189563
	FareFillBin         : 0.0559073094642
	Title_Mrs           : 0.0449742275771
	TicketAlphaPart     : 0.0440310311269
	Title_Miss          : 0.0406806667787
	Title_Master        : 0.0260531605083
Training accuracy: 0.844

It's overfitting more because of the TicketSize variable. (Forgot to remove it)

After removing it, it's still clearly overfitting but looks mostly the same as before.
Validation score 0.824 +/- 0.034, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 1}

FareFill qcut
-------------
qcut is generally better than fixed ranges I think.

From testing in ipython notebook it looks like 4 bins gives a good balance for correlation.

Ooops bug
---------
I wasn't applying the bin function on FareFill....

Validation score 0.824 +/- 0.032, Hyperparams {'max_features': 5, 'min_samples_split': 20, 'min_samples_leaf': 1}
Feature importances
	Title_Mr            : 0.358795301262
	SexNum              : 0.236140406327
	Pclass              : 0.190683131137
	FareFillBin         : 0.0837232786509
	TicketAlphaPart     : 0.0592992115209
	Title_Master        : 0.0311316932907
	Title_Miss          : 0.021469017801
	Title_Mrs           : 0.0187579600098

Doesn't seem to matter

FareFill qcut round 2
---------------------
4 bins seens good, it cuts them a little differently than the old ones 0-10, 10-20, 20-30, 30+

Validation score 0.816 +/- 0.035, Hyperparams {'max_features': 3, 'min_samples_split': 40, 'min_samples_leaf': 1}

Feature importances
	Title_Mr            : 0.311060094155
	SexNum              : 0.267142605502
	Pclass              : 0.175766572858
	FareFillBin         : 0.0748305161699
	Title_Miss          : 0.0584690635929
	Title_Mrs           : 0.048223681983
	TicketAlphaPart     : 0.0434659997668
	Title_Master        : 0.0210414659721

The learning curve looks weird... held-out error gets worse at the end with more data.

What about 5 bins? The learning curve looks dramatically better!
Validation score 0.828 +/- 0.036, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 2, 'min_samples_leaf': 3}

Feature importances
	Title_Mr            : 0.29114519712
	SexNum              : 0.187122682184
	Pclass              : 0.169025779537
	FareFillBin         : 0.135415494165
	Title_Miss          : 0.0681224528776
	TicketAlphaPart     : 0.0662293303083
	Title_Mrs           : 0.0572141367701
	Title_Master        : 0.0257249270371
Training accuracy: 0.846

What about 6 bins?
Looks terrible

FareFill qcut 5 bins
--------------------
Copy/pasted from above:

What about 5 bins? The learning curve looks dramatically better!
Validation score 0.828 +/- 0.036, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 2, 'min_samples_leaf': 3}

Feature importances
	Title_Mr            : 0.29114519712
	SexNum              : 0.187122682184
	Pclass              : 0.169025779537
	FareFillBin         : 0.135415494165
	Title_Miss          : 0.0681224528776
	TicketAlphaPart     : 0.0662293303083
	Title_Mrs           : 0.0572141367701
	Title_Master        : 0.0257249270371
Training accuracy: 0.846

Test acc 0.80383

FareFillBin as indicator features
---------------------------------
Validation score 0.828 +/- 0.036, Hyperparams {'max_features': None, 'min_samples_split': 20, 'min_samples_leaf': 2}
Feature importances
	Title_Mr            : 0.371277206412
	SexNum              : 0.240706486682
	Pclass              : 0.189044631562
	TicketAlphaPart     : 0.0463387740688
	FareFillBin_3       : 0.0412332409016
	Title_Master        : 0.0362336409706
	FareFillBin_4       : 0.0289650379829
	FareFillBin_1       : 0.0158231514825
	FareFillBin_0       : 0.0125241398598
	FareFillBin_2       : 0.0106743081239
	Title_Miss          : 0.00389081911385
	Title_Mrs           : 0.00328856283999

The graph is almost identical!

SibSp > 0
---------
It overfits more and validation error increases at the end.

Parch > 0
---------
Looks basically the same as trying SibSp.

Validation score 0.827 +/- 0.029, Hyperparams {'max_features': 8, 'min_samples_split': 30, 'min_samples_leaf': 1}

FamilySize > 1
--------------
Learning curve is better than with Parch but still doesn't look good.

Validation score 0.824 +/- 0.037, Hyperparams {'max_features': 0.5, 'min_samples_split': 2, 'min_samples_leaf': 3}

DeckNum
-------
The deck was important but not present too much. Let's see.

Introducing it overfits a ton more but regularization might bring that back down.
Validation score 0.823 +/- 0.034, Hyperparams {'max_features': 0.5, 'min_samples_split': 2, 'min_samples_leaf': 3}

Doesn't seem promising but I'll try submitting anyway.

Test acc 0.80861

Deck as indicators
------------------
Converting Deck to indicator variables and dropping Deck G and T cause they have too little data.

The graph isn't very stable but looks slightly better than the previous one.
Validation score 0.828 +/- 0.030, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 5, 'min_samples_leaf': 2}
Feature importances
	SexNum              : 0.206323441762
	Title_Mr            : 0.197157844396
	FareFillBin         : 0.126777417397
	Pclass              : 0.123385060321
	Title_Mrs           : 0.0737188660979
	TicketAlphaPart     : 0.0726766253841
	Title_Miss          : 0.0678493930691
	Deck_U              : 0.067652612273
	Title_Master        : 0.0254632187472
	Deck_E              : 0.0101059992689
	Deck_D              : 0.00796560662576
	Deck_B              : 0.00779484718337
	Deck_C              : 0.00746351536485
	Deck_A              : 0.00346141586714
	Deck_F              : 0.00220413624172

Test acc 0.81818 (best yet)

Did other family survive?
-------------------------
Validation score 0.842 +/- 0.036, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 2, 'min_samples_leaf': 3}

Feature importances
	Title_Mr            : 0.216238789727
	SexNum              : 0.187711482942
	TicketSurvival      : 0.126819658322
	Pclass              : 0.107987732469
	FareFillBin         : 0.078664422234
	Title_Miss          : 0.0658289817401
	Title_Mrs           : 0.0598042679416
	Deck_U              : 0.0522717704367
	TicketAlphaPart     : 0.0485635535582
	Title_Master        : 0.0187261079847
	Deck_E              : 0.0113140403409
	Deck_B              : 0.00811622255083
	Deck_C              : 0.00811172501428
	Deck_D              : 0.00666427513528
	Deck_A              : 0.00170904603652
	Deck_F              : 0.00146792356691

Test acc 0.81340 (very close!)

Did other family survive part 2
-------------------------------
I did a 3-way split but that might lead to overfitting.

Validation score 0.842 +/- 0.036, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 2, 'min_samples_leaf': 3}
Feature importances
	Title_Mr            : 0.211492462202
	SexNum              : 0.193574550502
	TicketSurvival      : 0.11049664737
	Pclass              : 0.109540796418
	FareFillBin         : 0.0826913118836
	Title_Mrs           : 0.0649365751623
	Title_Miss          : 0.0621038224369
	Deck_U              : 0.0551250518778
	TicketAlphaPart     : 0.0518475244596
	Title_Master        : 0.0209686993047
	Deck_E              : 0.0110297227903
	Deck_B              : 0.00890461524291
	Deck_D              : 0.00732139904263
	Deck_C              : 0.00665692629099
	Deck_A              : 0.00190917023693
	Deck_F              : 0.00140072477996

It's slightly more robust I think, submitting again.

Test acc 0.80383 (even worse!)

Did other family survive part 3
-------------------------------
Instead of checking if prob(survival) > 0, check if prob(survival) > 0.5

The learning curve looks slightly better but it's tough to say.

Validation score 0.838 +/- 0.033, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 5, 'min_samples_leaf': 2}
Feature importances
	Title_Mr            : 0.21174789242
	SexNum              : 0.189878832128
	Pclass              : 0.120761995964
	FareFillBin         : 0.117059427245
	TicketAlphaPart     : 0.0707498273083
	Title_Miss          : 0.0640248228397
	Title_Mrs           : 0.0575111406149
	Deck_U              : 0.0562563606083
	TicketSurvival      : 0.0419173561421
	Title_Master        : 0.0260426370823
	Deck_E              : 0.0141097908541
	Deck_B              : 0.0104015084829
	Deck_D              : 0.00751919691168
	Deck_C              : 0.00718327466433
	Deck_A              : 0.00250482025922
	Deck_F              : 0.00233111647604

The feature is used much much less which is what I want.

test acc 0.80383

Generalizing FareFill
=====================
The FareFill regressor is almost certainly overfitting. Let's try dropping all the little features.

Before
------
Fare hyperparameter tuning
Feature importances
	TicketSize          : 0.422876820229
	Pclass              : 0.383775854628
	CabinNum            : 0.0704656654573
	TicketAlphaPart     : 0.0446555919564
	DeckNum             : 0.0318466386355
	Parch               : 0.0224494637933
	SibSp               : 0.0119563134962
	Embarked_C          : 0.00345011534906
	Title_Miss          : 0.00283840056222
	Embarked_S          : 0.00195008625089
	Title_Dr            : 0.0013341042233
	Title_Mr            : 0.00115564253722
	Title_Mrs           : 0.000724340103577
	SexNum              : 0.000302812525584
	Title_Master        : 0.000149080920121
	Title_Military      : 2.65115199297e-05
	Embarked_Q          : 1.85699749614e-05
	Title_Sir           : 1.39705344702e-05
	Title_Lady          : 7.42717965424e-06
	Title_Rev           : 2.59012321608e-06
Validation score 0.885 +/- 0.059, Hyperparams {'max_features': 0.8, 'min_samples_split': 10, 'min_samples_leaf': 1}

Validation score 0.828 +/- 0.030, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 5, 'min_samples_leaf': 2}
Feature importances
	SexNum              : 0.206323441762
	Title_Mr            : 0.197157844396
	FareFillBin         : 0.126777417397
	Pclass              : 0.123385060321
	Title_Mrs           : 0.0737188660979
	TicketAlphaPart     : 0.0726766253841
	Title_Miss          : 0.0678493930691
	Deck_U              : 0.067652612273
	Title_Master        : 0.0254632187472
	Deck_E              : 0.0101059992689
	Deck_D              : 0.00796560662576
	Deck_B              : 0.00779484718337
	Deck_C              : 0.00746351536485
	Deck_A              : 0.00346141586714
	Deck_F              : 0.00220413624172

After
-----
Fare hyperparameter tuning
Feature importances
	TicketSize          : 0.360887081543
	Pclass              : 0.273428481131
	CabinNum            : 0.175758626446
	DeckNum             : 0.0686392292046
	TicketAlphaPart     : 0.0536471513711
	Parch               : 0.0393622753009
	SibSp               : 0.0282771550041
Validation score 0.893 +/- 0.056, Hyperparams {'max_features': 0.5, 'min_samples_split': 10, 'min_samples_leaf': 1}

So FareFill is correlating a touch better than it used to.

Validation score 0.827 +/- 0.030, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 5, 'min_samples_leaf': 2}
Feature importances
	SexNum              : 0.2084514096
	Title_Mr            : 0.198133421877
	FareFillBin         : 0.125957235255
	Pclass              : 0.121888245132
	Title_Mrs           : 0.0741492740886
	TicketAlphaPart     : 0.0725781132701
	Deck_U              : 0.0674181190726
	Title_Miss          : 0.0670276371001
	Title_Master        : 0.0258327978664
	Deck_E              : 0.0102744797719
	Deck_D              : 0.00767767492626
	Deck_C              : 0.00754900195891
	Deck_B              : 0.00752646234627
	Deck_A              : 0.00341538290595
	Deck_F              : 0.00212074482914

Overall CV score goes down and the FareFillBin feature importance goes down. The learning curves look identical so
I'm hoping for an identical score.

Test acc: 0.81818 (same as best)

Dropping SexNum
===============
The sex value is already captured in the title so it might help a tiny bit to drop it.

Validation score 0.824 +/- 0.029, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 1}
Feature importances
	Title_Mr            : 0.405125765096
	Pclass              : 0.137407469741
	FareFillBin         : 0.106101089646
	Title_Miss          : 0.0935691099185
	Title_Mrs           : 0.0898005756333
	Deck_U              : 0.0667069068971
	TicketAlphaPart     : 0.0549311722439
	Title_Master        : 0.014531785871
	Deck_E              : 0.0131866043783
	Deck_D              : 0.00553380248796
	Deck_C              : 0.00507307376047
	Deck_B              : 0.00355007653545
	Deck_A              : 0.00255334682659
	Deck_F              : 0.00192922096432

The learning curve seems to improve slightly. I do like that it picks a higher min samples split value.

Test acc: 0.79904 (wow way worse)

Dropping decks A, F
===================
These decks are uncommon and we're overfitting.

Validation score 0.824 +/- 0.029, Hyperparams {'max_features': 0.5, 'min_samples_split': 5, 'min_samples_leaf': 2}
Feature importances
	Title_Mr            : 0.252994058881
	SexNum              : 0.214232106259
	Pclass              : 0.140613275156
	FareFillBin         : 0.120597024915
	TicketAlphaPart     : 0.0762413434765
	Deck_U              : 0.0519293965434
	Title_Miss          : 0.0392111634075
	Title_Mrs           : 0.0357767704465
	Title_Master        : 0.0342061899009
	Deck_E              : 0.00979466398243
	Deck_B              : 0.00881119904348
	Deck_D              : 0.00798743186607
	Deck_C              : 0.00760537612227

The learning curve is about the same as the SexNum experiment but we clearly dropped the two least useful features.

Test acc: 0.80861 (not the best)

Dropping deck F
===============
Survival rates are a bit higher in Deck A I think so that feature should be useful. Most of the unlisted people
were in decks F and G I think so I'll try dropping just F.

Validation score 0.828 +/- 0.036, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 5, 'min_samples_leaf': 2}
Feature importances
	Title_Mr            : 0.254583523455
	SexNum              : 0.175261160755
	Pclass              : 0.127022098795
	FareFillBin         : 0.124379078307
	TicketAlphaPart     : 0.08213988704
	Title_Mrs           : 0.0596056159295
	Title_Miss          : 0.0577752065213
	Deck_U              : 0.0524297813549
	Title_Master        : 0.0260087187214
	Deck_E              : 0.0115795725636
	Deck_B              : 0.0091762365335
	Deck_D              : 0.00910936826172
	Deck_C              : 0.00727935264159
	Deck_A              : 0.00365039911963

The learning curve looks the same as the previous two.

Test acc 0.81340

Ticket number part, 3 bins
==========================
This clearly adds overfitting in the learning curve. And it worsens the validation score:
Validation score 0.820 +/- 0.042, Hyperparams {'max_features': 0.5, 'min_samples_split': 5, 'min_samples_leaf': 3}

Feature importances
	Title_Mr            : 0.277280395005
	SexNum              : 0.203985510335
	Pclass              : 0.111089081867
	FareFillBin         : 0.108471593271
	TicketNumPartBin    : 0.0617612990928
	TicketAlphaPart     : 0.0591149354841
	Deck_U              : 0.0553679711605
	Title_Master        : 0.0331570713903
	Title_Mrs           : 0.0321345441323
	Title_Miss          : 0.0281628546372
	Deck_E              : 0.00817860560064
	Deck_D              : 0.00745885545438
	Deck_C              : 0.00552233930483
	Deck_B              : 0.00457025295037
	Deck_A              : 0.00256902603441
	Deck_F              : 0.00117566428157

So I don't feel good about it. I also tried a test of dropping the ticket alpha part and only using the qcut3 num
part. That reduces overfitting but doesn't improve CV scores in the graph or numbers:
Validation score 0.819 +/- 0.036, Hyperparams {'max_features': 12, 'min_samples_split': 20, 'min_samples_leaf': 1}

It's possible that I can use more bins now because the ticket alpha part bins are gone, so I'll try 6 bins instead:
The learning curve looks more overfit than ticket alpha although it helps test slightly in the graph.
Validation score 0.817 +/- 0.036, Hyperparams {'max_features': 0.5, 'min_samples_split': 5, 'min_samples_leaf': 1}

Ticket Alpha Part fixing overfitting
====================================
There are many values of the ticket alpha that aren't common.
 
Delete everything after the slash
---------------------------------
This reduces the number of categories and there seems to be some correlation.

The learning curve again seems to learn faster but ends at about the same place.
Validation score 0.826 +/- 0.029, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 5, 'min_samples_leaf': 2}
Feature importances
	SexNum              : 0.210580565613
	Title_Mr            : 0.191805269537
	FareFillBin         : 0.124214477612
	Pclass              : 0.122058131909
	TicketAlphaPart     : 0.0759742836625
	Title_Mrs           : 0.0746829887002
	Title_Miss          : 0.0705189149462
	Deck_U              : 0.0648744171514
	Title_Master        : 0.0259587870559
	Deck_E              : 0.0110108549113
	Deck_B              : 0.00839434698228
	Deck_D              : 0.00751143333618
	Deck_C              : 0.00712897173474
	Deck_A              : 0.00308893363822
	Deck_F              : 0.00219762321039

Use indicators and drop the less common ones
--------------------------------------------
Dropping any with less than 10

Validation score 0.829 +/- 0.034, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 5, 'min_samples_leaf': 2}
Feature importances
	Title_Mr            : 0.222776604576
	SexNum              : 0.209152810184
	FareFillBin         : 0.113492962526
	Pclass              : 0.112348975669
	Title_Mrs           : 0.0730993267102
	Title_Miss          : 0.0668292022864
	Deck_U              : 0.056487287729
	Title_Master        : 0.0244739687445
	TicketAlphaPart_nan : 0.0222325506073
	TicketAlphaPart_PC  : 0.0147273986235
	Deck_B              : 0.0138802518694
	Deck_E              : 0.012744608545
	Deck_D              : 0.0106973772429
	TicketAlphaPart_CA  : 0.0105519321775
	TicketAlphaPart_STON: 0.00871274161103
	Deck_C              : 0.00841041422138
	TicketAlphaPart_W   : 0.00606879705104
	TicketAlphaPart_A   : 0.00521954610125
	Deck_A              : 0.00260658591969
	Deck_F              : 0.00244815249815
	TicketAlphaPart_SC  : 0.00169852006609
	TicketAlphaPart_SOTON: 0.00133998504069

The features were used a little bit. The graph looks a bit more jaggy on the test samples.

Dropping any under 5, less text normalization
---------------------------------------------
The graph looks slightly better.

Validation score 0.821 +/- 0.036, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 10, 'min_samples_leaf': 1}
Feature importances
	Title_Mr            : 0.236785839572
	SexNum              : 0.184341110226
	FareFillBin         : 0.119547502811
	Pclass              : 0.11163999539
	Deck_U              : 0.0601980578684
	Title_Mrs           : 0.058147999322
	Title_Miss          : 0.055938203894
	Title_Master        : 0.0240226642194
	TicketAlphaPart_nan : 0.0199762133285
	Deck_E              : 0.013344878211
	Deck_B              : 0.0126125233569
	Deck_D              : 0.012161090637
	TicketAlphaPart_STON/O: 0.0109377474212
	TicketAlphaPart_PC  : 0.0107279127617
	Deck_C              : 0.00919221730607
	TicketAlphaPart_C.A.: 0.00819390031701
	TicketAlphaPart_CA  : 0.0072252838949
	TicketAlphaPart_W./C.: 0.00711172903196
	TicketAlphaPart_CA. : 0.00563653585627
	TicketAlphaPart_A/5.: 0.00532402796898
	Deck_A              : 0.00421073519047
	Deck_F              : 0.00358245021609
	TicketAlphaPart_SOTON/O.Q.: 0.00301255477459
	TicketAlphaPart_C   : 0.00259488520467
	TicketAlphaPart_STON/O2.: 0.00232122437213
	TicketAlphaPart_S.O./P.P.: 0.00226994503224
	TicketAlphaPart_SC/PARIS: 0.00184487877996
	TicketAlphaPart_A/5 : 0.00165431397296
	TicketAlphaPart_F.C.C.: 0.00154189228297
	TicketAlphaPart_SC/Paris: 0.00118097836413
	TicketAlphaPart_S.O.C.: 0.00105689241631
	TicketAlphaPart_SOTON/OQ: 0.00104343431527
	TicketAlphaPart_SC/AH: 0.000362931052712
	TicketAlphaPart_A/4 : 0.000257450631199

Validation score is meh, let's try partial normalization (remove periods)

Validation score 0.824 +/- 0.032, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 5, 'min_samples_leaf': 2}
Feature importances
	Title_Mr            : 0.240679840391
	SexNum              : 0.206215532775
	Pclass              : 0.115950714941
	FareFillBin         : 0.104345018143
	Title_Miss          : 0.0742192585329
	Title_Mrs           : 0.0674874807248
	Deck_U              : 0.0464093643346
	Title_Master        : 0.0274757422102
	TicketAlphaPart_nan : 0.0154847625001
	TicketAlphaPart_PC  : 0.0144578878166
	Deck_B              : 0.0134568976442
	Deck_E              : 0.0132319641251
	Deck_D              : 0.0131595584599
	TicketAlphaPart_CA  : 0.0096100641489
	Deck_C              : 0.00833333118529
	TicketAlphaPart_STON/O: 0.00667215760635
	TicketAlphaPart_W/C : 0.0064026842871
	TicketAlphaPart_A/5 : 0.00273844035039
	Deck_A              : 0.00270534788287
	Deck_F              : 0.0023146295972
	TicketAlphaPart_SC/PARIS: 0.00226204285273
	TicketAlphaPart_STON/O2: 0.00127938614414
	TicketAlphaPart_C   : 0.00126127527278
	TicketAlphaPart_SOC : 0.00117847923008
	TicketAlphaPart_SOTON/OQ: 0.00116154174141
	TicketAlphaPart_FCC : 0.000679360512672
	TicketAlphaPart_A/4 : 0.000542882674727
	TicketAlphaPart_SO/PP: 0.000195079449904
	TicketAlphaPart_SC/AH: 8.92744657083e-05

In general the graph looks a tad better: training error gets closer to testing error a touch, testing error goes down a
touch, and variance in the testing error goes down a touch.

Test acc: 0.79904 (ugh)

On the bright side I got to add to my function to create indicator variables. All in all I made no progress on the scores
today.

Can slightly more aggressive regularization help?
-------------------------------------------------
Let's drop min samples split 5 and 10.

This wouldn't be represented in the graphs so I can't really compare. The cross-validation numbers
will go down because we're tuning fewer params to the data.

Validation score 0.820 +/- 0.031, Hyperparams {'max_features': 0.5, 'min_samples_split': 20, 'min_samples_leaf': 3}
Feature importances
	Title_Mr            : 0.290345720337
	SexNum              : 0.243246639129
	Pclass              : 0.136135951997
	FareFillBin         : 0.0958675950593
	Deck_U              : 0.0579645196574
	Title_Mrs           : 0.0552415313972
	TicketAlphaPart     : 0.040209026906
	Title_Miss          : 0.0313843168428
	Title_Master        : 0.0235480584565
	Deck_E              : 0.0101508261018
	Deck_D              : 0.00556842888996
	Deck_B              : 0.00376206741909
	Deck_C              : 0.00354171912137
	Deck_A              : 0.00166064709983
	Deck_F              : 0.00137295158547

Test acc: 0.80383

How about 10?
Validation score 0.821 +/- 0.032, Hyperparams {'max_features': 'sqrt', 'min_samples_split': 10, 'min_samples_leaf': 2}
Feature importances
	SexNum              : 0.211867929928
	Title_Mr            : 0.195436064591
	Pclass              : 0.12288923213
	FareFillBin         : 0.120921674026
	Title_Mrs           : 0.0809781683529
	Title_Miss          : 0.0762109241615
	Deck_U              : 0.0630536119916
	TicketAlphaPart     : 0.0629208517291
	Title_Master        : 0.027556483538
	Deck_E              : 0.0100728594007
	Deck_D              : 0.00821798386964
	Deck_B              : 0.00733278108714
	Deck_C              : 0.00706216929308
	Deck_A              : 0.00323354966585
	Deck_F              : 0.00224571623584

Test acc: 0.80861
This means that MSS 5 is actually important which amazes me.

MSS 3 gets
0.81340