import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
import new_utils as nu
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import top_k_accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        cc = {}
        for i,j in enumerate(counts):
            cc[uniq[i]] = j

        return {
            "class_counts": cc,  # Replace with actual class counts
            "num_classes": len(uniq),  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary

        answer = {}

        print("Part 3(A) - \n")
        #Xtrain, ytrain, Xtest, ytest = u.prepare_data()

        # Initialize the ShuffleSplit cross-validator
        #shuffle_split = ShuffleSplit(n_splits=5, random_state=42)

        # Initialize and train the classifier
        clf_log = LogisticRegression(max_iter=300, random_state=42)
        clf_log.fit(Xtrain,ytrain)

        k_values = [1, 2, 3, 4, 5]
        train_scores = []
        test_scores = []

        # Calculate top-k accuracy for each k
        for k in k_values:
            # Calculate top-k accuracy scores for the current k
            score_train_k = top_k_accuracy_score(ytrain, clf_log.predict_proba(Xtrain), k=k)  # Use a different variable name
            score_test_k = top_k_accuracy_score(ytest, clf_log.predict_proba(Xtest), k=k)    # Use a different variable name

            # Append the (k, score) tuples to the lists
            train_scores.append((k, score_train_k))
            test_scores.append((k, score_test_k))

            # Store the scores in the answer dictionary for each k
            answer[k] = {"score_train": score_train_k, "score_test": score_test_k}

        """
        # Plotting
        plt.plot(k_values, train_scores, label='Training Data')
        plt.plot(k_values, test_scores, label='Testing Data')
        plt.xlabel('Top k')
        plt.ylabel('Accuracy Score')
        plt.title('Top-k Accuracy Scores for Training and Testing Sets')
        plt.legend()
        plt.show()
        """

        #print("Training scores:", train_scores)
        #print("Testing scores:", test_scores)

        
        answer["clf"] = LogisticRegression(max_iter=300, random_state=42)
        answer["plot_k_vs_score_train"] = list(zip(k_values ,train_scores))
        answer["plot_k_vs_score_test"] = list(zip(k_values, test_scores))
        answer["text_rate_accuracy_change"] = "The rate of accuracy for testing data, increased with increase in the value of k"
        answer["text_is_topk_useful_and_why"] = "Yes, topk is useful because it measures the accuracy of a classifier's predictions when considering the top k predicted classes instead of just the most probable one."

        print(answer)
        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        print("Part 3(B) - \n")
        # X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = nu.filter_imbalanced_7_9s(X, y)
        Xtest, ytest = nu.filter_imbalanced_7_9s(Xtest, ytest)
        Xtrain_test = nu.scale_data(Xtrain)
        Xtest_test = nu.scale_data(Xtest)
        # Checking that the labels are integers
        ytrain_test = nu.scale_data_1(ytrain)
        ytest_test = nu.scale_data_1(ytest)
        print("3(B) - Are elements in Xtrain a floating point number and scaled between 0 to 1: " +str(Xtrain_test))
        print("3(B) - Are elements in a floating point number and scaled between 0 to 1: " +str(Xtest_test))
        print("3(B) - Are elements in ytrian an integer: " +str(ytrain_test))
        print("3(B) - Are elements in ytest an integer: " +str(ytest_test))
        answer = {}

        length_Xtrain1 = len(Xtrain)
        length_Xtest1 = len(Xtest)
        length_ytrain1 = len(ytrain)
        length_ytest1 = len(ytest)
        max_Xtrain1 = Xtrain.max()
        max_Xtest1 = Xtest.max()
        print(f"3(B) - Length of Xtrain, Xtest, ytrain, ytest is: {length_Xtrain1}, {length_Xtest1}, {length_ytrain1}, {length_ytest1}")
        print(f"3(B) - Max value of Xtrain and Xtest is: {max_Xtrain1}, {max_Xtest1}")
        answer["length_Xtrain"] = length_Xtrain1  # Number of samples
        answer["length_Xtest"] = length_Xtest1
        answer["length_ytrain"] = length_ytrain1
        answer["length_ytest"] = length_ytest1
        answer["max_Xtrain"] = max_Xtrain1
        answer["max_Xtest"] = max_Xtest1

        # Answer is a dictionary with the same keys as part 1.B

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Enter your code and fill the `answer` dictionary
        answer = {}

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        print("Part 3(D) - \n")

        #X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = nu.filter_imbalanced_7_9s(X, y)
        Xtest, ytest = nu.filter_imbalanced_7_9s(Xtest, ytest)
        
        def stratified_kfold():
            return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        
        def train_classifier_with_weighted_cv(Xtrain, ytrain, clf):
         # Compute class weights
            class_weights = compute_class_weight('balanced', classes=np.unique(ytrain), y=ytrain)
            class_weight_dict = dict(enumerate(class_weights))
            #print("Class weights:", class_weight_dict)
            scoring_1 = {'accuracy': 'accuracy', 'f1_score': make_scorer(f1_score, average='macro'),'precision': make_scorer(precision_score, average='macro'),'recall': make_scorer(recall_score, average='macro')}
            #cross-validation
            scores = cross_validate(clf, Xtrain, ytrain, cv=stratified_kfold(), scoring=scoring_1, fit_params={'sample_weight': [class_weight_dict[y] for y in ytrain]})
            # Print the mean and std of scores
            u.print_cv_result_dict(scores)
            return scores, class_weight_dict
    
        # Train SVC with cross-validation using weighted loss function
        clf_svc_weighted = SVC(random_state=42)
        scores_svc_weighted, class_weight_dict_outside = train_classifier_with_weighted_cv(Xtrain, ytrain, clf_svc_weighted)

        
        print("The weights are: \n")
        print(class_weight_dict_outside)
        

        # Train the classifier on all training data
        clf_svc_weighted.fit(Xtrain, ytrain)

        # Plot confusion matrix for the test data
        y_pred_svc_train1 = clf_svc_weighted.predict(Xtrain)
        y_pred_svc_test1 = clf_svc_weighted.predict(Xtest)
        cm_svc_train1 = confusion_matrix(ytrain, y_pred_svc_train1)
        cm_svc_test1 = confusion_matrix(ytest, y_pred_svc_test1)
        print("Confusion matrix for training data: \n")
        print(cm_svc_train1)
        print("Confusion matrix for testing data:")
        print(cm_svc_test1)

        scores_dict = scores_svc_weighted

        score_values_svc2={}
        for key,array in scores_dict.items():
            if(key=='test_accuracy'):
                score_values_svc2['mean_accuracy'] = array.mean()
                score_values_svc2['std_accuracy'] = array.std()
            if(key=='test_f1_score'):
                score_values_svc2['mean_f1'] = array.mean()
                score_values_svc2['std_f1'] = array.std()
            if(key=='test_precision'):
                score_values_svc2['mean_precision'] = array.mean()
                score_values_svc2['std_precision'] = array.std()
            if(key=='test_recall'):
                score_values_svc2['mean_recall'] = array.mean()
                score_values_svc2['std_recall'] = array.std()

        answer["scores"] = score_values_svc2
        answer["cv"] = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        answer["clf"] = SVC(random_state=42)
        answer["class_weights"] = class_weight_dict_outside
        answer["confusion_matrix_train"] = confusion_matrix(ytrain, y_pred_svc_train1)
        answer["confusion_matrix_test"] = confusion_matrix(ytest, y_pred_svc_test1)
        answer["explain_purpose_of_class_weights"] = "Class weights are used to address class imbalance in classification problems"
        answer["explain_performance_difference"] = "Using class weights in the SVM classifier results in improved overall accuracy, F1 score, and recall, indicating better performance, particularly in correctly identifying positive instances. However, there's a slight decrease in precision when using class weights. This trade-off suggests that while class weights help in better identifying minority class instances, there may be a slight increase in false positives."

        print(answer)
        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
