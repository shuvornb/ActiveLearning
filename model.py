import random
import numpy as np
from sklearn.linear_model import LogisticRegression
import math

ITERATION_COUNT = 50
BATCH_SIZE = 10


def train_model_and_get_accuracy(train_data, train_labels, test_data, test_labels):
    model = LogisticRegression(max_iter=500)
    model.fit(train_data, train_labels.ravel())
    return model, model.score(test_data, test_labels)


def apply_random_sampling(train_data, train_labels, train_data_un, train_labels_un):
    for i in range(0, BATCH_SIZE):
        size = len(train_data_un) - 1
        index = random.randint(0, size)
        train_data = np.append(train_data, [train_data_un[index]], axis=0)
        train_data_un = np.delete(train_data_un, index, axis=0)
        train_labels = np.append(train_labels, [train_labels_un[index]], axis=0)
        train_labels_un = np.delete(train_labels_un, index, axis=0)
    return train_data, train_labels, train_data_un, train_labels_un


def apply_uncertainty_based_sampling(train_data, train_labels, train_data_un, train_labels_un, model):
    predictions = model.predict_proba(train_data_un)
    entropy_scores = []
    for i in predictions:
        entropy = 0
        for x in i:
            entropy = entropy + x * math.log(x, 2)
        entropy = entropy * -1
        entropy_scores.append(entropy)

    for i in range(0, BATCH_SIZE):
        max_value = max(entropy_scores)
        index = entropy_scores.index(max_value)
        entropy_scores.pop(index)
        train_data = np.append(train_data, [train_data_un[index]], axis=0)
        train_data_un = np.delete(train_data_un, index, axis=0)
        train_labels = np.append(train_labels, [train_labels_un[index]], axis=0)
        train_labels_un = np.delete(train_labels_un, index, axis=0)
    return train_data, train_labels, train_data_un, train_labels_un


def apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un, algo_type):
    accuracy = []
    print(algo_type)
    print("===========")
    for i in range(0, ITERATION_COUNT):

        # train the model and get the accuracy
        lr_model, acc = train_model_and_get_accuracy(train_data, train_labels, test_data, test_labels)
        print("Iteration: ", i+1, " Accuracy: ", acc)
        accuracy.append(acc)

        # update train data
        if algo_type == "RANDOM":
            train_data, train_labels, train_data_un, train_labels_un = apply_random_sampling(train_data, train_labels,
                                                                                             train_data_un,
                                                                                             train_labels_un)
        else:
            train_data, train_labels, train_data_un, train_labels_un = apply_uncertainty_based_sampling(train_data,
                                                                                                        train_labels,
                                                                                                        train_data_un,
                                                                                                        train_labels_un,
                                                                                                        lr_model)
    return accuracy
