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