#!/usr/bin/env python3

from sklearn import metrics
import itertools
import matplotlib.pyplot as plt
import json


class Metrics:

    def plot_confusion_matrix(self, y_true, y_pred, normalize=True):
        matrix = metrics.confusion_matrix(y_true, y_pred)
        if normalize:
            matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        cmap = plt.get_cmap('Blues')
        plt.imshow(matrix, interpolation='nearest', cmap=cmap)
        plt.title("Test set Results")
        plt.colorbar()

        value_threshold = matrix.max() / 1.5 if normalize else matrix.max() / 2

        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(matrix[i, j]),
                         horizontalalignment="center",
                         color="white" if matrix[i, j] > value_threshold else "black")
            else:
                plt.text(j, i, "{:,}".format(matrix[i, j]),
                         horizontalalignment="center",
                         color="white" if matrix[i, j] > value_threshold else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def test_results(self, args, y_true, y_pred, loss):
        precision, recall, fmeasure, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average="macro")
        acc = (y_pred == y_true).mean()
        json_results = {
            "acc": acc,
            "loss": loss,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accumulation": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "precision": precision,
            "recall": recall,
            "f-measure": fmeasure
        }
        return json.dumps(json_results)