import numpy as np
import pandas as pd
import os

from sktime.classification.hybrid import HIVECOTEV1
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest


def stand_sca(data):
    new_data=(data-data.mean())/data.std()
    return new_data


def load_data():
    positive_train_path = "/Users/albertmilagro/Desktop/LS-data/hg/train"  # 合格样品训练集
    cc_train_path = "/Users/albertmilagro/Desktop/LS-data/cc/train"  # 粗糙样品训练集
    hh_train_path = "/Users/albertmilagro/Desktop/LS-data/hh/train"  # 划痕样品训练集
    ld_train_path = "/Users/albertmilagro/Desktop/LS-data/ld/train"  # 漏度样品训练集

    files = os.listdir(positive_train_path)  # 读入文件夹
    positive_count = len([f for f, v in enumerate(files) if v.split(".")[1] == "txt"])

    files = os.listdir(cc_train_path)  # 读入文件夹
    cc_count = len([f for f, v in enumerate(files) if v.split(".")[1] == "txt"])

    files = os.listdir(hh_train_path)  # 读入文件夹
    hh_count = len([f for f, v in enumerate(files) if v.split(".")[1] == "txt"])

    files = os.listdir(ld_train_path)  # 读入文件夹
    ld_count = len([f for f, v in enumerate(files) if v.split(".")[1] == "txt"])

    train_size = positive_count + cc_count + hh_count + ld_count

    X_train = pd.DataFrame(pd.Series, columns=["val"], index=np.arange(train_size))
    y_train = np.concatenate((np.zeros(positive_count), np.ones(cc_count),
                              2 * np.ones(hh_count), 3* np.ones(ld_count)), axis=0)  # 标签为hg0 + cc1 + hh2 + ld3

    # load hg data
    os.chdir(positive_train_path)
    count = 0  # position in X_train dataframe
    for dirName, dirs, files in os.walk("."):
        for f in files:
            if f.split('.')[1] != 'txt':
                continue
            data = pd.read_csv(f, header=None, dtype="float", sep="\s+", encoding='unicode-escape').iloc[:, 1]
            X_train.iloc[count, 0] = stand_sca(data)
            count += 1
    # load cc data
    os.chdir(cc_train_path)
    for dirName, dirs, files in os.walk("."):
        for f in files:
            if f.split('.')[1] != 'txt':
                continue
            data = pd.read_csv(f, header=None, dtype="float", sep="\s+", encoding='unicode-escape').iloc[:, 1]
            X_train.iloc[count, 0] = stand_sca(data)
            count += 1
    # load hh data
    os.chdir(hh_train_path)
    for dirName, dirs, files in os.walk("."):
        for f in files:
            if f.split('.')[1] != 'txt':
                continue
            data = pd.read_csv(f, header=None, dtype="float", sep="\s+", encoding='unicode-escape').iloc[:, 1]
            X_train.iloc[count, 0] = stand_sca(data)
            count += 1

    # load ld data
    os.chdir(ld_train_path)
    for dirName, dirs, files in os.walk("."):
        for f in files:
            if f.split('.')[1] != 'txt':
                continue
            data = pd.read_csv(f, header=None, dtype="float", sep="\s+", encoding='unicode-escape').iloc[:, 1]
            X_train.iloc[count, 0] = stand_sca(data)
            count += 1

    return X_train, y_train


def train(X_train, y_train):
    clf = HIVECOTEV1(
        stc_params={
            "estimator": RotationForest(n_estimators=3),
            "n_shapelet_samples": 500,
            "max_shapelets": 20,
            "batch_size": 100,
        },
        tsf_params={"n_estimators": 10},
        rise_params={"n_estimators": 10},
        cboss_params={"n_parameter_samples": 25, "max_ensemble_size": 5},
    )
    clf.fit(X_train, y_train)
    return clf


def test(clf):
    test_path = "/Users/albertmilagro/Desktop/LS-data/ld/data"  # 测试数据
    count = 0
    os.chdir(test_path)
    files = os.listdir(test_path)  # 读入文件夹
    test_count = len([f for f, v in enumerate(files) if v.split(".")[1] == "txt"])
    files.sort()
    X_test = pd.DataFrame(pd.Series, columns=["val"], index=np.arange(test_count))
    file_list = []
    for f in files:
        if f.split('.')[1] != 'txt':
            continue
        data = pd.read_csv(f, header=None, dtype="float", sep="\s+", encoding='unicode-escape').iloc[:, 1]
        X_test.iloc[count, 0] = stand_sca(data)
        count += 1
        file_list.append(f)
    pred_result = clf.predict(X_test)
    for i in range(0, len(file_list)):
        if pred_result[i] == 0:
            print("合格", " ", file_list[i])
        elif pred_result[i] == 1:
            print("粗糙", " ", file_list[i])
        elif pred_result[i] == 2:
            print("划痕", " ", file_list[i])
        else:
            print("漏镀", " ", file_list[i])
        print("--------------")


def learn():
    X_train, y_train = load_data()
    print("----------------------")
    print("Data loaded, begin to learn")
    clf = train(X_train, y_train)
    print("Learning Finished")
    print("----------------------")
    test(clf)

