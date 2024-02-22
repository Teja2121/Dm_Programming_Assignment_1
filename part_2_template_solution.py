# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
import new_utils as nu
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    cross_val_score,
    KFold
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        answer = {}
        # Enter your code and fill the `answer`` dictionary

        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        Xtrain_test = nu.scale_data(Xtrain)
        Xtest_test = nu.scale_data(Xtest)
        # Checking that the labels are integers
        ytrain_test = nu.scale_data_1(ytrain)
        ytest_test = nu.scale_data_1(ytest)

        # Calculate lengths of datasets and labels
        length_Xtrain = Xtrain.shape[0]
        length_Xtest = Xtest.shape[0]
        length_ytrain = ytrain.shape[0]
        length_ytest = ytest.shape[0]   
        # Calculate maximum values in datasets
        max_Xtrain = Xtrain.max()
        max_Xtest = Xtest.max()
        # Calculate the number of classes and class counts for the training and testing set
        unique_classes_train, class_count_train = np.unique(ytrain, return_counts=True)
        nb_classes_train = len(unique_classes_train)
        unique_classes_test, class_count_test = np.unique(ytest, return_counts=True)
        nb_classes_test = len(unique_classes_test)

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set
        answer["nb_classes_train"] = nb_classes_train
        answer["nb_classes_test"] = nb_classes_test
        answer["class_count_train"] = class_count_train
        answer["class_count_test"] = class_count_test
        answer["length_Xtrain"] = length_Xtrain
        answer["length_Xtest"] = length_Xtest
        answer["length_ytrain"] = length_ytrain
        answer["length_ytest"] = length_ytest
        answer["max_Xtrain"] = max_Xtrain
        answer["max_Xtest"] = max_Xtest
        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        ytrain = ytest = np.zeros([1], dtype="int")

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        # Enter your code and fill the `answer`` dictionary
        answer = {}

        X, y, Xtest, ytest = u.prepare_data()

        for ntrain, ntest in zip(ntrain_list, ntest_list):
            x_r = X[0:ntrain, :]
            y_r = y[0:ntrain]
            Xtest_r = Xtest[0:ntest, :]
            ytest_r = ytest[0:ntest]
            scores2b1c = u.train_simple_classifier_with_cv(Xtrain= x_r, ytrain= y_r, clf=DecisionTreeClassifier(random_state=42), cv=KFold(n_splits=5, shuffle = True, random_state=42))
            scores2b1d = u.train_simple_classifier_with_cv(Xtrain=x_r, ytrain=y_r, clf=DecisionTreeClassifier(random_state=42), cv=ShuffleSplit(n_splits=5, random_state=42))

            clf_partF = LogisticRegression(max_iter=300, random_state=42)
            cv_partF = ShuffleSplit(n_splits=5, random_state=52)
            clf_partF.fit(x_r, y_r)
            y_prediction_train_F = clf_partF.predict(x_r)
            y_prediction_test_F = clf_partF.predict(Xtest_r)

            scores_train_F = accuracy_score(y_r, y_prediction_train_F)
            scores_test_F = accuracy_score(ytest_r, y_prediction_test_F)

            cross_val_scores_F = cross_val_score(clf_partF, x_r, y_r, cv=cv_partF)
            mean_cv_accuracy_F = cross_val_scores_F.mean()

            conf_mat_train = confusion_matrix(y_r, y_prediction_train_F)
            conf_mat_test = confusion_matrix(ytest_r, y_prediction_test_F)

            scores2b1f = u.train_simple_classifier_with_cv(Xtrain=x_r, ytrain=y_r, clf=LogisticRegression(max_iter=300, random_state=42), cv=ShuffleSplit(n_splits=5, random_state=42))

            mean_fit_time_1c = np.mean(scores2b1d['fit_time'])
            std_fit_time_1c = np.std(scores2b1d['fit_time'])
            mean_accuracy_1c = np.mean(scores2b1d['test_score'])
            std_accuracy_1c = np.std(scores2b1d['test_score'])
            mean_fit_time_1d = np.mean(scores2b1d['fit_time'])
            std_fit_time_1d = np.std(scores2b1d['fit_time'])
            mean_accuracy_1d = np.mean(scores2b1d['test_score'])
            std_accuracy_1d = np.std(scores2b1d['test_score'])


            unique_classes, class_count_train = np.unique(y_r, return_counts=True)
            class_count_train_list = class_count_train.tolist()
            unique_classes, class_count_test = np.unique(ytest_r, return_counts=True)
            class_count_test_list = class_count_test.tolist()

            answer[ntrain] = {
                "partC" : {
                    "scores" : {"mean_fit_time" : mean_fit_time_1c, "std_fit_time" : std_fit_time_1c, "mean_accuracy" : mean_accuracy_1c, "std_accuracy" : std_accuracy_1c
                                },
                    "clf" : DecisionTreeClassifier(random_state=42), 
                    "cv" : KFold(n_splits=5, shuffle = True, random_state=42) 
                            },
                "partD" : {
                    "scores" : {"mean_fit_time" : mean_fit_time_1d, "std_fit_time" : std_fit_time_1d, "mean_accuracy" : mean_accuracy_1d, "std_accuracy" : std_accuracy_1d
                                },
                    "clf" : DecisionTreeClassifier(random_state=42), 
                    "cv" : ShuffleSplit(n_splits=5, random_state=42)             
                },
                "partF" : {
                    "scores_train_F" : scores_train_F, 
                    "scores_test_F" : scores_test_F, 
                    "mean_cv_accuracy_F" : mean_cv_accuracy_F,
                    "clf" : clf_partF,
                    "cv" : cv_partF,
                    "conf_mat_train" : conf_mat_train,
                    "conf_mat_test" : conf_mat_test
                },
                "ntrain" : ntrain,
                "ntest" : ntest,
                "class_count_train" : class_count_train_list,
                "class_count_test" : class_count_test_list

                        }
            
        
        return answer
        
        
