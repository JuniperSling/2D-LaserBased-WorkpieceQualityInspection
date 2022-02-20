# 切换系统需要修改52到85行的分隔符'\\' 与'/'
import sys
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import os
from PyQt5 import QtCore,  QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QInputDialog, QMessageBox
from MainWindow import Ui_DataClassifier
from LoginWindow import Ui_LoginWindow
from sktime.classification.hybrid import HIVECOTEV1
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
import matplotlib.pyplot as plt


# 数据标准化
def stand_sca(data):
    new_data=(data-data.mean())/data.std()
    return new_data


def kick_bad(x, y):  # 剔除坏点
    left = y.mean() - 2 * y.std()
    delete = []
    for i in range(0, len(y)):
        if y[i] < left:
            delete.append(i)
    x = np.delete(x, delete)
    y = np.delete(y, delete)
    return x, y


# 登入弹窗
class Login(QMainWindow, Ui_LoginWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.time_label.setText(str(datetime.today().date()) + "   " + str(datetime.today().time())[0:8])

    def confirm(self):
        # 记录日志
        self.worker_name = self.name_line.text()
        self.worker_code = self.code_line.text()
        self.sampinfo = self.sampInfo_line.text()
        self.create_logs()
        window.show()
        self.close()

    # 创建日志
    def create_logs(self):
        # 创建日志文件夹
        if not os.path.exists(os.getcwd() + "/Logs"):
            os.makedirs(os.getcwd() + "/Logs")
        day = str(datetime.today().date())  # 获取操作日期
        time = str(datetime.today().time())[0:8]  # 获取操作时间

        # 写入当日txt日志,打开主程序窗体
        self.log_path = os.getcwd() + "/Logs/" + day + ".csv"
        log = pd.DataFrame({'登陆时间': [day + " " + time], '操作人': [self.worker_name], '样品信息': [self.sampinfo]})
        if not os.path.exists(self.log_path):
            log.to_csv(self.log_path, index=False, encoding='utf_8_sig')
        else:
            log.to_csv(self.log_path, index=False, header=False, encoding='utf_8_sig', mode='a+')


# 主窗口
class PyQtMainEntry(QMainWindow, Ui_DataClassifier):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 载入初始UI
        # 获取范例图片目录
        if not os.path.exists(os.getcwd() + "/Sample_pics"):
            os.makedirs(os.getcwd() + "/Sample_pics")
        self.sample_pic_path = os.getcwd() + "/Sample_pics"

        # 获取分类器默认保存目录
        if not os.path.exists(os.getcwd() + "/Classifiers"):
            os.makedirs(os.getcwd() + "/Classifiers")
        self.clf_path = os.getcwd() + "/Classifiers"

        # 获取结果默认保存目录
        if not os.path.exists(os.getcwd() + "/Results"):
            os.makedirs(os.getcwd() + "/Results")
        self.result_file = os.getcwd() + "/Results"

    # 上传训练数据文件夹
    def upload_train(self):
        self.train_path = QFileDialog.getExistingDirectory()
        if self.train_path == "": return
        self.upload_train_label.setText("路径：" + self.train_path)  # 更新路径标签

        self.train_files = os.listdir(self.train_path)  # 读入文件夹中的所有文件
        self.train_files = [f for f in self.train_files if (f.split(".")[-1] == "txt" and len(f.split("-", 1)[0]) > 0)]  # 保留txt格式带标签文件
        self.train_clf_butt.setEnabled(True)

    # 上传待分类数据文件夹
    def upload_test(self):
        self.test_path = QFileDialog.getExistingDirectory()
        if self.test_path == "": return
        self.upload_test_label.setText("路径：" + self.test_path)  # 更新路径标签
        self.test_label.setText("请选择工作模式...")

        self.test_files = os.listdir(self.test_path)  # 读入文件夹中的所有文件
        self.test_files = [f for f in self.test_files if f.split(".")[1] == "txt"]  # 保留后缀为txt格式的文件

        self.test_butt.setEnabled(True)
        self.realtime_test_butt.setEnabled(True)

    # 更新分类种类下拉框
    def update_type_list(self, path):
        file = open(path, "r", encoding="utf_8_sig")
        self.type_list = []
        for line in file.readlines():
            self.type_list.append(line.split('\n')[0])
        file.close()  # 将文件关闭

    # 上传本地分类器
    def upload_clf(self):
        self.local_clf_path, ok_pressed = QFileDialog.getOpenFileName(self, "选择分类器", filter="Clf Files(*.pickle)")
        if not ok_pressed: return
        if not os.path.exists(self.local_clf_path.rsplit(".", 1)[0] + ".txt"):
            msg = QMessageBox(text='找不到同名text后缀文件，加载失败！')
            msg.exec_()
            return
        with open(self.local_clf_path, 'rb') as file:
            self.clf = pickle.load(file)
        file.close()
        self.update_type_list(self.local_clf_path.rsplit(".", 1)[0] + ".txt")

        self.upload_clf_label.setText("路径：" + self.local_clf_path)
        self.clf_label.setText("当前分类器：" + self.local_clf_path)

        self.upload_test_butt.setEnabled(True)  # 允许上传待分类数据
        self.types_combobox.setEnabled(True)
        self.types_combobox.clear()
        for type in self.type_list:
            self.types_combobox.addItem(type)

    # 训练新的分类器
    def train(self):
        if len(self.upload_train_label.text()) < 4:
            return
        X_train = pd.DataFrame(pd.Series, columns=["val"], index=np.arange(len(self.train_files)))  # 初始化样本集X
        y_label = []
        self.clf = HIVECOTEV1(stc_params={"estimator": RotationForest(n_estimators=3), "n_shapelet_samples": 500,
                                          "max_shapelets": 20, "batch_size": 100}, tsf_params={"n_estimators": 10},
                              rise_params={"n_estimators": 10}, cboss_params={"n_parameter_samples": 25,
                                                                              "max_ensemble_size": 5},)  # 初始化分类器

        os.chdir(self.train_path)  # 切换工作目录到训练集

        self.type_list = ["HG"]  # 下标数对应种类名字
        pos = 0  # position in X_train dataframe
        for f in self.train_files:
            # 存入一个标准化后的文件数据
            X_train.iloc[pos, 0] = stand_sca(pd.read_csv(f, header=None, dtype="float", sep="\s+", encoding='unicode-escape').iloc[:, 1])
            file_type = f.split("-")[0]  # 合格（HG）:0, 不合格：1、2、3...
            # 存入标签y
            if file_type in self.type_list:
                y_label.append(self.type_list.index(file_type))
            else:
                self.type_list.append(file_type)
                y_label.append(self.type_list.index(file_type))
            pos += 1

        y_train = np.array(y_label)
        self.clf_label.setText("当前分类器：正在训练分类器...")
        self.clf.fit(X_train, y_train)
        self.clf_label.setText("当前分类器：训练完毕！")
        clf_name, _ = QInputDialog.getText(self, "保存分类器", "请给分类器起名")

        # 创建文件夹保存分类器
        if not os.path.exists(self.clf_path):
            os.makedirs(self.clf_path)
        # 保存分类器
        with open(self.clf_path + "/" + clf_name + '.pickle', 'wb') as file:
            pickle.dump(self.clf, file)
        file.close()
        # 写入分类种类信息
        file = open(self.clf_path + "/" + clf_name + '.txt', 'w', encoding='utf_8_sig')
        for type in self.type_list:
            file.write(str(type))
            file.write('\n')
        file.close()

        self.clf_label.setText("当前分类器：" + self.clf_path + "/" + clf_name + '.pickle')

        self.upload_test_butt.setEnabled(True)  # 允许上传待分类数据
        self.update_type_list(self.clf_path + "/" + clf_name + '.txt')
        self.types_combobox.setEnabled(True)
        self.types_combobox.clear()
        for type in self.type_list:
            self.types_combobox.addItem(type)

    # 开始批量处理
    def test(self):
        self.mode = 'all_at_once'
        # self.father_path = self.test_path.rsplit('/', 1)[0]  # 获取父目录地址
        result_name, ok_pressed = QInputDialog.getText(self, "保存结果", "请输入文件名", text=login_window.sampinfo
                                                                                    + '-' + self.mode)
        if not ok_pressed: return
        self.result_path = self.result_file + "/" + result_name + ".csv"
        self.test_label.setText("全部处理完成！\n结果路径：" + self.result_path)

        X_test = pd.DataFrame(pd.Series, columns=["val"], index=np.arange(len(self.test_files)))  # 初始化待训练数据表

        os.chdir(self.test_path)
        pos = 0
        for f in self.test_files:
            data = pd.read_csv(f, header=None, dtype="float", sep="\s+", encoding='unicode-escape').iloc[:, 1]
            X_test.iloc[pos, 0] = stand_sca(data)
            pos += 1
        # 预测结果
        y_test = self.clf.predict(X_test)
        prob = self.clf.predict_proba(X_test)
        predick_type_list = [self.type_list[f] for f in y_test]
        prob_list = [str(100 * max(f))[0:2] + "%" for f in prob]

        # 保存为csv文件
        data = pd.DataFrame({'预测种类': predick_type_list, '文件名': self.test_files, '可信度': prob_list})
        data.to_csv(self.result_path, index=False, encoding='utf_8_sig')
        # 模式设置， 控件激活
        self.manual_confirm_butt.setEnabled(True)
        self.manual_combobox.setEnabled(True)
        self.manual_combobox.clear()
        for type in self.type_list:
            self.manual_combobox.addItem(type)
        # 合格计数显示
        self.hg_count = 0
        for y in y_test:
            if y == 0: self.hg_count += 1
        self.bhg_count = len(y_test) - self.hg_count
        self.hg_count_label.setText("合格数：" + str(self.hg_count))
        self.bhg_count_label.setText("不合格数：" + str(self.bhg_count))
        # 记录导出日志
        self.export_log(result_name, self.result_path)
        self.manual()  # 手动处理

    # 开始实时处理
    def realtime_test(self):
        self.mode = 'real_time'
        self.type_label.setText("分析结果：")
        self.possibility_label.setText("可信度：")
        # self.father_path = self.test_path.rsplit('/', 1)[0]  # 获取父目录地址
        result_name, ok_pressed = QInputDialog.getText(self, "保存结果", "请输入文件名", text=login_window.sampinfo + "-" + self.mode)
        if not ok_pressed: return
        threshold, ok_pressed = QInputDialog.getInt(self, "设置人工调整阈值", "请输入可信度阈值(0~100之间的整数)，"
                                                                      "低于此值提醒人工判定", value=0, min=0, max=100)
        if not ok_pressed: threshold = 0
        self.threshold = threshold
        self.threshold_label.setText("当前可信度阈值：" + str(threshold))
        self.result_path = self.result_file + "/" + result_name + ".csv"
        self.file_watcher = QtCore.QFileSystemWatcher(self)

        # 获取初时刻文件
        self.test_files = os.listdir(self.test_path)  # 读入文件夹中的所有文件
        self.test_files = [f for f in self.test_files if f.split(".")[1] == "txt"]  # 保留后缀为txt格式的文件

        # 创建监视器
        self.file_watcher.addPath(self.test_path)
        self.file_watcher.directoryChanged.connect(self.file_changed)

        self.realtime_test_butt.setEnabled(False)
        self.test_butt.setEnabled(False)
        self.upload_test_butt.setEnabled(False)
        self.upload_clf_butt.setEnabled(False)
        self.upload_train_butt.setEnabled(False)
        self.manual_combobox.setEnabled(False)
        self.manual_confirm_butt.setEnabled(False)
        self.test_label.setText("正在监测文件改动...\n结果路径：" + self.result_path)
        self.manual_pic.setText("还没有新增txt文件...")
        self.halt_realtime_test_butt.setEnabled(True)
        self.manual_combobox.setEnabled(True)
        self.manual_combobox.clear()
        for type in self.type_list:
            self.manual_combobox.addItem(type)
        self.hg_count = 0
        self.bhg_count = 0
        self.hg_count_label.setText("合格数" + str(self.hg_count))
        self.bhg_count_label.setText("不合格数" + str(self.bhg_count))

    # 根据下拉框展示样例图片
    def show_samp_pic(self):
        if not os.path.exists(self.sample_pic_path + "/" + self.types_combobox.currentText() + '.jpg'):
            self.sample_pic.setText("没有找到图片，在Sample_pics文件夹上传同名jpg图片吧")
        else:
            self.sample_pic.setPixmap(
                QtGui.QPixmap(self.sample_pic_path + "/" + self.types_combobox.currentText() + '.jpg').
                    scaled(self.sample_pic.width(), self.sample_pic.height()))

    # 检测到文件改动信号
    def file_changed(self):
        # 读取此刻文件列表
        new_files = os.listdir(self.test_path)
        # 找到新增文件
        files_added = [f for f in new_files if ((f.split(".")[1] == "txt") and (f not in self.test_files))]
        self.test_files = [f for f in new_files if f.split(".")[1] == "txt"]
        # 考虑实际情况，实时模式一次信号只读取一个文件，不支持多文件拖入
        if len(files_added) > 0:
            self.manual_combobox.setEnabled(True)
            self.manual_confirm_butt.setEnabled(True)
            self.detele_data_butt.setEnabled(True)
            self.pic_name_label.setText(files_added[0])
            X_test = pd.DataFrame(pd.Series, columns=["val"], index=[0])  # 初始化待训练数据表
            os.chdir(self.test_path)
            X_test.iloc[0, 0] = stand_sca(pd.read_csv(files_added[0], header=None,
                                                      dtype="float", sep="\s+", encoding='unicode-escape').iloc[:, 1])
            # 预测结果
            y_test = self.clf.predict(X_test)
            prob = self.clf.predict_proba(X_test)
            predict_type_list = [self.type_list[f] for f in y_test]
            prob_list = [str(100 * max(f))[0:2] + "%" for f in prob]
            # 追加写入csv文件
            data = pd.DataFrame({'预测种类': predict_type_list, '文件名': [files_added[0]], '可信度': prob_list})
            if not os.path.exists(self.result_path):
                data.to_csv(self.result_path, index=False, encoding='utf_8_sig', mode='a+')
            else:
                data.to_csv(self.result_path, index=False, encoding='utf_8_sig', mode='a+', header=False)
            # 绘图模块
            x = np.loadtxt(self.test_path + "/" + files_added[0], usecols=0, encoding='unicode_escape').ravel()
            y = np.loadtxt(self.test_path + "/" + files_added[0], usecols=1, encoding='unicode_escape').ravel()
            x, y = kick_bad(x, y)
            plt.plot(x, y)
            plt.scatter(x, y, marker='x', c='r')
            plt.savefig(self.sample_pic_path + "/temp.jpg")
            plt.clf()
            self.manual_pic.setPixmap(
                QtGui.QPixmap(self.sample_pic_path + '/temp.jpg').scaled(self.manual_pic.width(),
                                                                         self.manual_pic.height()))
            self.type_label.setText("分析结果：" + predict_type_list[0])
            self.types_combobox.setCurrentText(predict_type_list[0])
            self.possibility_label.setText("可信度：" + prob_list[0])
            # 合格计数
            if y_test[0] == 0:
                self.hg_count += 1
                self.hg_count_label.setText("合格数" + str(self.hg_count))
            else:
                self.bhg_count += 1
                self.bhg_count_label.setText("不合格数" + str(self.bhg_count))
            # 可信度较低弹窗提醒
            if int(prob_list[0].split("%")[0]) < self.threshold:
                msg_box = QMessageBox(QMessageBox.Warning, '提示', '可信度较低，请复核！')
                msg_box.exec_()
            # 记录导出日志
            if self.hg_count + self.bhg_count == 1:  # 第一个数据
                self.export_log(self.result_path.rsplit("/", 1)[1].split(".")[0], self.result_path)
            else:  # 不是第一个数据
                log = pd.read_csv(self.export_log_path, encoding='utf_8_sig')
                filename = log.iloc[-1, 3]
                path = log.iloc[-1, 4]
                log_new = log.drop([len(log) - 1])
                log_new.to_csv(self.export_log_path, index=False, encoding='utf_8_sig')  # 删去最后一行
                self.export_log(filename, path)  # 重新添加最后一行

    # 停止实时处理
    def halt_realtime_test(self):
        self.file_watcher.removePath(self.test_path)
        self.realtime_test_butt.setEnabled(True)
        self.test_butt.setEnabled(True)
        self.upload_test_butt.setEnabled(True)
        self.upload_clf_butt.setEnabled(True)
        self.upload_train_butt.setEnabled(True)
        self.test_label.setText("已停止实时监测")
        self.halt_realtime_test_butt.setEnabled(False)
        self.manual_confirm_butt.setEnabled(False)
        self.manual_combobox.setEnabled(False)
        self.detele_data_butt.setEnabled(False)

    # 样例图片同步变动
    def set_samp_combobox(self):
        self.types_combobox.setCurrentText(self.manual_combobox.currentText())

    # 批量处理人工调整阶段绘图显示模块
    def show_pic(self):
        if len(self.index_to_be_tuned) < 1:
            self.manual_pic.setText("没有符合条件的数据，尝试提高阈值！")
            return

        x = np.loadtxt(self.test_path + "/" + self.out_put_data.iloc[self.index_to_be_tuned[0], 1], usecols=0, encoding='unicode_escape').ravel()
        y = np.loadtxt(self.test_path + "/" + self.out_put_data.iloc[self.index_to_be_tuned[0], 1], usecols=1, encoding='unicode_escape').ravel()
        x, y = kick_bad(x, y)
        plt.plot(x, y)
        plt.scatter(x, y, marker='x', c='r')
        plt.savefig(self.sample_pic_path + "/temp.jpg")
        plt.clf()
        self.manual_pic.setPixmap(
            QtGui.QPixmap(self.sample_pic_path + '/temp.jpg').scaled(self.manual_pic.width(), self.manual_pic.height()))
        self.type_label.setText("分析结果：" + self.out_put_data.iloc[self.index_to_be_tuned[0], 0])
        self.types_combobox.setCurrentText(self.out_put_data.iloc[self.index_to_be_tuned[0], 0])
        self.possibility_label.setText("可信度：" + self.out_put_data.iloc[self.index_to_be_tuned[0], 2])

    # 初始化批处理手工调整
    def manual(self):
        threshold, ok_pressed = QInputDialog.getInt(self, "设置人工调整阈值", "请输入可信度阈值(0~100之间的整数)，"
                                                                      "低于此值可以人工判定", value=0, min=0, max=100)
        if not ok_pressed: threshold = 0
        self.threshold = threshold
        self.out_put_data = pd.read_csv(self.result_path, encoding='utf_8_sig')  # 加载原数据
        self.index_to_be_tuned = []  # 待处理数据下标位置
        # 加载需要处理的文件下标
        for i in range(0, len(self.out_put_data)):
            if int(self.out_put_data.iloc[i, 2].split("%")[0]) < self.threshold:
                self.index_to_be_tuned.append(i)
        self.tune_count = str(len(self.index_to_be_tuned))
        self.show_pic()
        filename = ""
        if len(self.index_to_be_tuned) > 0:
            filename = self.out_put_data.iloc[self.index_to_be_tuned[0], 1]
        self.pic_name_label.setText("1/" + self.tune_count + " " + filename)
        self.threshold_label.setText("当前可信度阈值：" + str(self.threshold) + "%")

    # 批量处理的按钮修改动作
    def change(self, new_label):
        if len(self.index_to_be_tuned) == 0: return
        else:
            # 更新合格计数器
            if new_label != self.out_put_data.iloc[self.index_to_be_tuned[0], 0]:  # 修改标签和原来的不一样
                if new_label == self.type_list[0]:  # 不合格修改为合格
                    self.hg_count += 1
                    self.bhg_count -= 1
                else:  # 合格修改为不合格
                    if self.out_put_data.iloc[self.index_to_be_tuned[0], 0] == self.type_list[0]:
                        self.hg_count -= 1
                        self.bhg_count += 1
                self.hg_count_label.setText("合格数：" + str(self.hg_count))
                self.bhg_count_label.setText("不合格数：" + str(self.bhg_count))

            # 更新csv
            self.out_put_data.iloc[self.index_to_be_tuned[0], 0] = new_label
            self.out_put_data.iloc[self.index_to_be_tuned[0], 2] = "100%"
            self.out_put_data.to_csv(self.result_path, encoding='utf_8_sig', index=False)
            self.index_to_be_tuned.pop(0)

            # 更新导出日志
            log = pd.read_csv(self.export_log_path, encoding='utf_8_sig')
            filename = log.iloc[-1, 3]
            path = log.iloc[-1, 4]
            log_new = log.drop([len(log) - 1])  # 删去最后一行
            log_new.to_csv(self.export_log_path, index=False, encoding='utf_8_sig')
            self.export_log(filename, path)  # 重新添加最后一行

        if len(self.index_to_be_tuned) > 0:
            # 展示下一个数据
            self.show_pic()
            self.pic_name_label.setText(str(int(self.tune_count) - len(self.index_to_be_tuned) + 1) + "/" + self.tune_count
                                        + " " + self.out_put_data.iloc[self.index_to_be_tuned[0], 1])
        else:
            self.pic_name_label.setText(self.tune_count + "/" + self.tune_count)
            self.manual_pic.setText("已经完成全部人工矫正！")
            self.manual_confirm_butt.setEnabled(False)
            self.manual_combobox.setEnabled(False)

    # 实时处理的按钮修改动作
    def realtime_change(self, new_label):
        self.out_put_data = pd.read_csv(self.result_path, encoding='utf_8_sig')  # 加载原数据
        # 更新合格计数器
        if new_label != self.out_put_data.iloc[-1, 0]:  # 修改标签和原来的不一样
            if new_label == self.type_list[0]:  # 不合格修改为合格
                self.hg_count += 1
                self.bhg_count -= 1
            else:  # 合格修改为不合格
                if self.out_put_data.iloc[-1, 0] == self.type_list[0]:
                    self.hg_count -= 1
                    self.bhg_count += 1
            self.hg_count_label.setText("合格数：" + str(self.hg_count))
            self.bhg_count_label.setText("不合格数：" + str(self.bhg_count))
        # 更新csv
        self.out_put_data.iloc[-1, 0] = new_label
        self.out_put_data.iloc[-1, 2] = "100%"
        self.out_put_data.to_csv(self.result_path, encoding='utf_8_sig', index=False)
        self.manual_confirm_butt.setEnabled(False)
        self.manual_combobox.setEnabled(False)
        self.detele_data_butt.setEnabled(False)
        # 更新导出日志
        log = pd.read_csv(self.export_log_path, encoding='utf_8_sig')
        filename = log.iloc[-1, 3]
        path = log.iloc[-1, 4]
        log_new = log.drop([len(log) - 1])
        log_new.to_csv(self.export_log_path, index=False, encoding='utf_8_sig')  # 删去最后一行
        self.export_log(filename, path)  # 重新添加最后一行

    # 实时处理的删除功能
    def delete(self):
        self.out_put_data = pd.read_csv(self.result_path, encoding='utf_8_sig')  # 加载原数据
        # 更新合格计数
        label = self.out_put_data.iloc[-1, 0]
        if label == self.type_list[0]:  # 删去合格数据
            self.hg_count -= 1
        else:  # 删去不合格数据
            self.bhg_count -= 1
        self.hg_count_label.setText("合格数：" + str(self.hg_count))
        self.bhg_count_label.setText("不合格数：" + str(self.bhg_count))
        # 删除表格末行数据
        file_to_be_deleted = self.out_put_data.iloc[-1, 1]
        data_new = self.out_put_data.drop([len(self.out_put_data) - 1])
        data_new.to_csv(self.result_path, index=False, encoding='utf_8_sig')
        # 删除数据文件
        os.remove(self.test_path + "/" + file_to_be_deleted)

        self.manual_pic.setText("删除成功！等待新的数据...")
        self.detele_data_butt.setEnabled(False)
        self.manual_confirm_butt.setEnabled(False)
        self.manual_combobox.setEnabled(False)
        # 修改导出日志
        log = pd.read_csv(self.export_log_path, encoding='utf_8_sig')
        filename = log.iloc[-1, 3]
        path = log.iloc[-1, 4]
        log_new = log.drop([len(log) - 1])
        log_new.to_csv(self.export_log_path, index=False, encoding='utf_8_sig')  # 删去最后一行
        self.export_log(filename, path)  # 重新添加最后一行

    # 修改后确认按钮重载
    def manual_confirm(self):
        if self.mode == "all_at_once":
            # 修改为combo 的标签
            self.change(self.manual_combobox.currentText())

        if self.mode == "real_time":
            self.realtime_change(self.manual_combobox.currentText())

    # 生成导出日志
    def export_log(self, filename, path):
        self.export_log_path = login_window.log_path.rsplit(".", 1)[0] + "-导出记录" + ".csv"
        if (self.bhg_count + self.hg_count) <= 0:
            hg_rate = 0
        else:
            hg_rate = self.hg_count / (self.bhg_count + self.hg_count) * 100
        log = pd.DataFrame({'导出时间': [str(datetime.today().date()) + " " + str(datetime.today().time())[0:8]],
                            '操作人': [login_window.worker_name], '样品信息': [login_window.sampinfo],
                            '文件名': filename, '目录': path, '模式':self.mode, '合格数量': self.hg_count,
                            '不合格数量': self.bhg_count, '合格率': hg_rate})
        if not os.path.exists(self.export_log_path):
            log.to_csv(self.export_log_path, index=False, encoding='utf_8_sig')
        else:
            log.to_csv(self.export_log_path, index=False, header=False, encoding='utf_8_sig', mode='a+')


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    login_window = Login()
    login_window.show()
    sys.exit(app.exec_())
