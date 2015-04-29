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