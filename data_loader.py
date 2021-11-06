from scipy.io import loadmat
from sklearn import preprocessing


def get_mr_data_1():
    train_data = loadmat("data/MindReading/trainingMatrix_MindReading1.mat")
    train_labels = loadmat("data/MindReading/trainingLabels_MindReading_1.mat")
    test_data = loadmat("data/MindReading/testingMatrix_MindReading1.mat")
    test_labels = loadmat("data/MindReading/testingLabels_MindReading1.mat")
    train_data_un = loadmat("data/MindReading/unlabeledMatrix_MindReading1.mat")
    train_labels_un = loadmat("data/MindReading/unlabeledLabels_MindReading_1.mat")

    train_data = train_data['trainingMatrix']
    train_labels = train_labels['trainingLabels']
    test_data = test_data['testingMatrix']
    test_labels = test_labels['testingLabels']
    train_data_un = train_data_un['unlabeledMatrix']
    train_labels_un = train_labels_un['unlabeledLabels']

    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    train_data_un = preprocessing.MinMaxScaler().fit_transform(train_data_un)

    return train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un


def get_mr_data_2():
    train_data = loadmat("data/MindReading/trainingMatrix_MindReading2.mat")
    train_labels = loadmat("data/MindReading/trainingLabels_MindReading_2.mat")
    test_data = loadmat("data/MindReading/testingMatrix_MindReading2.mat")
    test_labels = loadmat("data/MindReading/testingLabels_MindReading2.mat")
    train_data_un = loadmat("data/MindReading/unlabeledMatrix_MindReading2.mat")
    train_labels_un = loadmat("data/MindReading/unlabeledLabels_MindReading_2.mat")

    train_data = train_data['trainingMatrix']
    train_labels = train_labels['trainingLabels']
    test_data = test_data['testingMatrix']
    test_labels = test_labels['testingLabels']
    train_data_un = train_data_un['unlabeledMatrix']
    train_labels_un = train_labels_un['unlabeledLabels']

    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    train_data_un = preprocessing.MinMaxScaler().fit_transform(train_data_un)

    return train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un


def get_mr_data_3():
    train_data = loadmat("data/MindReading/trainingMatrix_MindReading3.mat")
    train_labels = loadmat("data/MindReading/trainingLabels_MindReading_3.mat")
    test_data = loadmat("data/MindReading/testingMatrix_MindReading3.mat")
    test_labels = loadmat("data/MindReading/testingLabels_MindReading3.mat")
    train_data_un = loadmat("data/MindReading/unlabeledMatrix_MindReading3.mat")
    train_labels_un = loadmat("data/MindReading/unlabeledLabels_MindReading_3.mat")

    train_data = train_data['trainingMatrix']
    train_labels = train_labels['trainingLabels']
    test_data = test_data['testingMatrix']
    test_labels = test_labels['testingLabels']
    train_data_un = train_data_un['unlabeledMatrix']
    train_labels_un = train_labels_un['unlabeledLabels']

    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    train_data_un = preprocessing.MinMaxScaler().fit_transform(train_data_un)

    return train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un


def get_mmi_data_1():
    train_data = loadmat("data/MMI/trainingMatrix_1.mat")
    train_labels = loadmat("data/MMI/trainingLabels_1.mat")
    test_data = loadmat("data/MMI/testingMatrix_1.mat")
    test_labels = loadmat("data/MMI/testingLabels_1.mat")
    train_data_un = loadmat("data/MMI/unlabeledMatrix_1.mat")
    train_labels_un = loadmat("data/MMI/unlabeledLabels_1.mat")

    train_data = train_data['trainingMatrix']
    train_labels = train_labels['trainingLabels']
    test_data = test_data['testingMatrix']
    test_labels = test_labels['testingLabels']
    train_data_un = train_data_un['unlabeledMatrix']
    train_labels_un = train_labels_un['unlabeledLabels']

    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    train_data_un = preprocessing.MinMaxScaler().fit_transform(train_data_un)

    return train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un


def get_mmi_data_2():
    train_data = loadmat("data/MMI/trainingMatrix_2.mat")
    train_labels = loadmat("data/MMI/trainingLabels_2.mat")
    test_data = loadmat("data/MMI/testingMatrix_2.mat")
    test_labels = loadmat("data/MMI/testingLabels_2.mat")
    train_data_un = loadmat("data/MMI/unlabeledMatrix_2.mat")
    train_labels_un = loadmat("data/MMI/unlabeledLabels_2.mat")

    train_data = train_data['trainingMatrix']
    train_labels = train_labels['trainingLabels']
    test_data = test_data['testingMatrix']
    test_labels = test_labels['testingLabels']
    train_data_un = train_data_un['unlabeledMatrix']
    train_labels_un = train_labels_un['unlabeledLabels']

    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    train_data_un = preprocessing.MinMaxScaler().fit_transform(train_data_un)

    return train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un


def get_mmi_data_3():
    train_data = loadmat("data/MMI/trainingMatrix_3.mat")
    train_labels = loadmat("data/MMI/trainingLabels_3.mat")
    test_data = loadmat("data/MMI/testingMatrix_3.mat")
    test_labels = loadmat("data/MMI/testingLabels_3.mat")
    train_data_un = loadmat("data/MMI/unlabeledMatrix_3.mat")
    train_labels_un = loadmat("data/MMI/unlabeledLabels_3.mat")

    train_data = train_data['trainingMatrix']
    train_labels = train_labels['trainingLabels']
    test_data = test_data['testingMatrix']
    test_labels = test_labels['testingLabels']
    train_data_un = train_data_un['unlabeledMatrix']
    train_labels_un = train_labels_un['unlabeledLabels']

    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    train_data_un = preprocessing.MinMaxScaler().fit_transform(train_data_un)

    return train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un
