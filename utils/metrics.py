import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Metrics:
    labels_gestures = ['down', 'flip_table', 'idle', 'pinch', 'rotate_left','rotate_right', 'spin', 'spread', 'swipe_left', 'swipe_right','up']
    #labels_gestures = ['idle', 'rotate', 'swipe_left', 'swipe_right']

    # X_predicted = neural_net.predict(X_validate)
    # x_interpreted = helpers.interpret_predictions(X_predicted)
    # y_interpreted = helpers.interpret_predictions(y_validate)
    def __init__(self, X_predicted, y, x_interpreted, y_interpreted):
        self.prediction = X_predicted
        self.Y = y
        self.x_interpreted = x_interpreted
        self.y_interpreted = y_interpreted
        self.set_all(X_predicted, y, x_interpreted, y_interpreted)
        

    def set_all(self, X_predicted, y, x_interpreted, y_interpreted):
        self.set_accuracy(x_interpreted, y_interpreted)
        self.set_true_false_pos_neg(X_predicted, y)
        self.set_per_class_accuracy(x_interpreted, y_interpreted)
        self.set_precision()
        self.set_recall()
        self.set_f1_score()
        self.set_macro_precision()
        self.set_macro_recall()
        self.set_macro_f1_score()

    def print_metrics(self):
        print(f"Accuracy: {self.accuracy}")
        # visualization for true false pos neg
        self.visualize_true_false_pos_neg()
        # confusion matrix
        self.print_confusion_matrix(self.x_interpreted, self.y_interpreted)
        # print metrics per class
        for i in range(len(self.labels_gestures)):
            print(f"\n{self.labels_gestures[i]}: ")
            print(f"Accuracy: {self.per_class_accuracy[i]}")
            print(f"Precision: {self.precision[i]}")
            print(f"Recall: {self.recall[i]}")
            print(f"F1-Score: {self.f1[i]}")

        print(f"\nMacro Precision: {self.macro_precision}")
        print(f"Macro Recall: {self.macro_recall}")
        print(f"Macro F1-Score: {self.macro_f1}")

    def set_accuracy(self, H, Y):
        if H.ndim > 1 and Y.ndim > 1:
            correct = 0
            total = len(H)
            for i in range(len(H)):
                if(Y[i] == H[i]):
                    correct += 1
            self.accuracy = (correct/total)
        else:
            self.accuracy = np.mean(H == Y)

    def get_accuracy(self):
        return self.accuracy

    def set_true_false_pos_neg(self, H, Y):
        if H.ndim > 1 and Y.ndim > 1:
            self.true_positives = []
            self.false_positives = []
            self.true_negatives = []
            self.false_negatives = []
            for i in range(H.shape[1]): # hier gehe ich die einzelnen Gesten durch
                # [: ,i] => so kann man auf die jeweilige Spalte zugreifen
                self.true_positives.append((H[: ,i].round() == 1) & (Y[: ,i] == 1))
                self.false_positives.append((H[: ,i].round() == 1) & (Y[: ,i] == 0))
                self.true_negatives.append((H[: ,i].round() == 0) & (Y[: ,i] == 0))
                self.false_negatives.append((H[: ,i].round() == 0) & (Y[: ,i] == 1))
        else: 
            self.true_positives = (H.round() == 1) & (Y == 1)
            self.false_positives = (H.round() == 1) & (Y == 0)
            self.true_negatives = (H.round() == 0) & (Y == 0)
            self.false_negatives = (H.round() == 0) & (Y == 1)
        return self.true_positives, self.false_positives, self.true_negatives, self.false_negatives

    def get_true_false_pos_neg(self):
        return  self.true_positives, self.false_positives, self.true_negatives, self.false_negatives

    def visualize_true_false_pos_neg(self):
        for i in range(self.prediction.shape[1]):
            confusion_matrix = np.array([[self.true_negatives[i].sum(), self.false_negatives[i].sum()],
                                [self.false_positives[i].sum(), self.true_positives[i].sum()]]).astype(int)
            fig, ax = plt.subplots(figsize=(5, 4))

            labels = ["true negatives\n %d", "false negatives\n %d", "false positives\n %d", "true positives\n %d"]
            label_values = np.array([l % v for l,v in zip(labels, confusion_matrix.flat)]).reshape(2,2)

            sns.heatmap(confusion_matrix, annot=label_values, fmt="", ax=ax)
            ax.set_xlabel("ground truth")
            ax.set_ylabel("predicted")
            ax.set_title(f"confusion matrix for {self.labels_gestures[i]}")

    def print_confusion_matrix(self,H, Y):
        actual = pd.Series(Y, name='Actual')
        predicted = pd.Series(H, name='Predicted')
        confusion_matrix = pd.crosstab(actual, predicted)
        # print(confusion_matrix)
        fig, ax = plt.subplots(figsize=(8,12))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", ax=ax, xticklabels=self.labels_gestures, yticklabels=self.labels_gestures)
    

    def set_per_class_accuracy(self, H, Y):
        classes = np.unique(Y)
        self.per_class_accuracy = [((H == Y)[Y == klass]).mean() for klass in classes]

    def get_per_class_accuracy(self):
        return self.per_class_accuracy

    def set_precision(self):
        self.precision = []
        for i in range(self.prediction.shape[1]):
            self.precision.append(np.sum(self.true_positives[i]) / (np.sum(self.true_positives[i]) + np.sum(self.false_positives[i])))
    
    def get_precision(self):
        return self.precision

    def set_recall(self):
        self.recall = []
        for i in range(self.prediction.shape[1]):
            self.recall.append(np.sum(self.true_positives[i]) / (np.sum(self.true_positives[i]) + np.sum(self.false_negatives[i])))
    
    def get_recall(self):
        return self.recall

    def set_f1_score(self):
        self.f1 = []
        for i in range(self.prediction.shape[1]):
            self.f1.append(2 * (self.precision[i] * self.recall[i])/(self.precision[i] + self.recall[i]))
    
    def get_f1_score(self):
        return self.f1
    
    def set_macro_precision(self):
        self.macro_precision = np.mean(self.precision)

    def get_macro_precision(self):
        return self.macro_precision
    
    def set_macro_recall(self):
        self.macro_recall = np.mean(self.recall)

    def get_macro_recall(self):
        return self.macro_recall
    
    def set_macro_f1_score(self):
        self.macro_f1 = np.mean(self.f1)

    def get_macro_f1_score(self):
        return self.macro_f1