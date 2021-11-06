import data_loader
import model
import matplotlib.pyplot as plt
import sys

ITERATION_COUNT = 50

if __name__ == "__main__":

    data_type = sys.argv[1]

    if data_type == "MMI":
        # applying random sampling on MMI data
        train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un = data_loader.get_mmi_data_1()
        acc1 = model.apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un,
                                           train_labels_un,
                                           "RANDOM")

        train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un = data_loader.get_mmi_data_2()
        acc2 = model.apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un,
                                           train_labels_un,
                                           "RANDOM")

        train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un = data_loader.get_mmi_data_3()
        acc3 = model.apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un,
                                           train_labels_un,
                                           "RANDOM")

        x1 = []
        y1 = []
        for i in range(0, ITERATION_COUNT):
            x1.append(i + 1)
            y1.append((acc1[i] + acc2[i] + acc3[i]) / 3)

        # applying uncertainty sampling on MMI data
        train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un = data_loader.get_mmi_data_1()
        acc1 = model.apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un,
                                           train_labels_un,
                                           "UNCERTAINTY")

        train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un = data_loader.get_mmi_data_2()
        acc2 = model.apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un,
                                           train_labels_un,
                                           "UNCERTAINTY")

        train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un = data_loader.get_mmi_data_3()
        acc3 = model.apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un,
                                           train_labels_un,
                                           "UNCERTAINTY")

        x2 = []
        y2 = []
        for i in range(0, ITERATION_COUNT):
            x2.append(i + 1)
            y2.append((acc1[i] + acc2[i] + acc3[i]) / 3)

        plt.plot(x1, y1, label="Random Sampling")
        plt.plot(x2, y2, label="Uncertainty Based Sampling")
        plt.xlabel('Iteration Count')
        plt.ylabel('Accuracy')
        plt.title('Active Learning - ' + data_type)
        plt.legend()
        plt.show()

    elif data_type == "MR":
        # applying random sampling on MR data
        train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un = data_loader.get_mr_data_1()
        acc1 = model.apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un,
                                           train_labels_un,
                                           "RANDOM")

        train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un = data_loader.get_mr_data_2()
        acc2 = model.apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un,
                                           train_labels_un,
                                           "RANDOM")

        train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un = data_loader.get_mr_data_3()
        acc3 = model.apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un,
                                           train_labels_un,
                                           "RANDOM")

        x1 = []
        y1 = []
        for i in range(0, ITERATION_COUNT):
            x1.append(i + 1)
            y1.append((acc1[i] + acc2[i] + acc3[i]) / 3)

        # applying uncertainty sampling on MR data
        train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un = data_loader.get_mr_data_1()
        acc1 = model.apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un,
                                           train_labels_un,
                                           "UNCERTAINTY")

        train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un = data_loader.get_mr_data_2()
        acc2 = model.apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un,
                                           train_labels_un,
                                           "UNCERTAINTY")

        train_data, train_labels, test_data, test_labels, train_data_un, train_labels_un = data_loader.get_mr_data_3()
        acc3 = model.apply_active_learning(train_data, train_labels, test_data, test_labels, train_data_un,
                                           train_labels_un,
                                           "UNCERTAINTY")

        x2 = []
        y2 = []
        for i in range(0, ITERATION_COUNT):
            x2.append(i + 1)
            y2.append((acc1[i] + acc2[i] + acc3[i]) / 3)

        plt.plot(x1, y1, label="Random Sampling")
        plt.plot(x2, y2, label="Uncertainty Based Sampling")
        plt.xlabel('Iteration Count')
        plt.ylabel('Accuracy')
        plt.title('Active Learning - ' + data_type)
        plt.legend()
        plt.show()

    else:
        print("Invalid command.")
