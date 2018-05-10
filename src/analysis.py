import configparser
import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import _pickle
import simplejson
from sklearn.metrics import confusion_matrix

# config parameters
config = configparser.ConfigParser()
config.read('config.ini')


def plot_confusion_matrix(y_true, y_pred):
    print('plot confusion matrix start: ', end='')
    for label in zip(y_true, y_pred):
        if (label[0] not in [0, 1]) or (label[1] not in [0, 1]):
            y_true.remove(label[0])
            y_pred.remove(label[1])

    # compute confusion matrix
    cnf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # configuration
    np.set_printoptions(precision=2)
    labels = ['benign', 'malware']
    norm_flag = True
    plot_title = 'Confusion matrix'
    cmap = plt.cm.Blues

    if norm_flag:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # plotting start
    plt.figure()
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(plot_title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    # plt.xticks(tick_marks, labels, rotation=90)
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # information about each block's value
    fmt = '.7f' if norm_flag else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    ## insert legend information
    # import matplotlib.patches as mpatches
    # patches = [mpatches.Patch(color='white', label='G{num} = {group}'.format(num=i+1, group=labels[i])) for i in range(len(labels))]
    # plt.legend(handles=patches, bbox_to_anchor=(-0.60, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print('--plot confusion matrix finish--')
    pass


def get_kaspersky_info(file_lists):
    print('# of file lists: {}'.format(len(file_lists)))
    with open(kasp_label_json, 'r') as f:
        kasp_label_dict = simplejson.load(f)

    result = list()
    for file_path in file_lists:
        md5 = os.path.splitext(os.path.split(file_path)[-1])[0]
        date = kasp_label_dict[md5]['collected_date']
        group_name = kasp_label_dict[md5]['kasp_grp_name']

        feature_hash_vector = _pickle.load(open(file_path, 'rb'))
        zero_cnt_3, zero_cnt_4, zero_cnt_5 = 0,0,0
        each_fh_vec_size = int(len(feature_hash_vector) / 3)
        for i in range(each_fh_vec_size):
            if feature_hash_vector[i] == 0:
                zero_cnt_3 += 1
            if feature_hash_vector[i+each_fh_vec_size] == 0:
                zero_cnt_4 += 1
            if feature_hash_vector[i+each_fh_vec_size*2] == 0:
                zero_cnt_5 += 1
        result.append([md5, date, group_name, zero_cnt_3, zero_cnt_4, zero_cnt_5])
    return result


def analyze_fn(file_lists, y_true, y_pred, model_num):  # 오탐 데이터 분석, if y_true = 1 and y_pred = 0 -> false negative
    number_of_data = len(file_lists)
    print(number_of_data, len(y_true), len(y_pred))
    if not (number_of_data == len(y_true) and number_of_data == len(y_pred)):
        print('evaluate error! {} {} {}'.format(number_of_data, len(y_true), len(y_pred)))
        return False

    search_file_lists = list()
    for i in range(len(file_lists)):
        if y_true[i] == 1 and y_pred[i] == 0:
            search_file_lists.append(file_lists[i])
    print('search finish')

    result = get_kaspersky_info(search_file_lists)

    with open('analyze{}.csv'.format(model_num), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(
            ['md5', 'collected_date', 'kasp_grp_name', 'fh_3_zero',
             'fh_4_zero', 'fh_5_zero', 'raw size', 'ops size'])
        for line in result:
            wr.writerow(line)
    pass


if __name__ == '__main__':
    with open('result3.pickle', 'rb') as f:
        a = _pickle.load(f)
        b = _pickle.load(f)
        c = _pickle.load(f)
        pass
    plot_confusion_matrix(b, c)
    pass