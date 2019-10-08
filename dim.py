import h5py
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.models import Model
from sklearn import svm
from keras.models import load_model
import numpy as np

from keras.datasets import mnist

from picture_edit.ave_smooth import ave_smooth

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_x, img_y = 28, 28

x_train_smooth = ave_smooth(x_train)
x_test_smooth = ave_smooth(x_test)

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

x_train_smooth = x_train_smooth.reshape(x_train.shape[0], img_x, img_y, 1)
x_test_smooth = x_test_smooth.reshape(x_test.shape[0], img_x, img_y, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train_smooth = x_train_smooth.astype('float32')
x_test_smooth = x_test_smooth.astype('float32')

x_train /= 255
x_test /= 255

x_train_smooth /= 255
x_test_smooth /= 255

class1 = []  # 单分类器训练集
for i in range(0, y_train.size):
    if y_train[i] == 3:
        class1.append(x_train[i])
class1 = np.array(class1)
print("单分类器训练集shape：")
print(x_train.shape)

test1 = []  # 单分类器检验集
for i in range(0, y_test.size):
    if y_test[i] == 3:
        test1.append(x_test[i])
test1 = np.array(test1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("加载第两个模型：")
# model = load_model("F://sundries/Models/number.h5")
# middle = load_model("F://sundries/Models/middle.h5")

model = load_model("F://sundries/Models/number_kern.h5")
middle = load_model("F://sundries/Models/middle_kern.h5")

print("构建源自x_test的测试集")
class2 = []
one_class_test = x_test_smooth
re = model.predict(one_class_test)
re2 = np.argmax(re, axis=1)     # 把独热编码变换一下
y_train_10 = np.argmax(y_test, axis=1)
for i in range(y_train_10.size):
    if re2[i] == 3 and y_train_10[i] != 3:
        class2.append(x_test[i])
class2 = np.array(class2)
print(class2.shape)

print("构建源自x_train的测试集")
class3 = []
one_class_test = x_train_smooth
re = model.predict(one_class_test)
re2 = np.argmax(re, axis=1)     # 把独热编码变换一下
y_train_10 = np.argmax(y_test, axis=1)
for i in range(y_train_10.size):
    if re2[i] == 3 and y_train_10[i] != 3:
        class3.append(x_test[i])
class3 = np.array(class3)
print(class3.shape)


print("构建SVM训练集")
dense1_output = middle.predict(class1)
print(dense1_output.shape)
print(dense1_output)

print("OneClass 建模")
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=1e-4)
print("训练OneClass")
clf.fit(dense1_output)

print("测试集from x_test")
SVM_test = middle.predict(class2)
re3 = clf.decision_function(SVM_test)
print(re3)


print("测试集from x_train")
SVM_test = middle.predict(class3)
re3 = clf.decision_function(SVM_test)
print(re3)


print("正例 from x_test")
SVM_test = middle.predict(test1)
print(SVM_test)

print("正例 from x_train 也就是过训练集")
re3 = clf.decision_function(SVM_test)
re = clf.decision_function(dense1_output)
print(re)