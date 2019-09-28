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

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_x, img_y = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

class1 = []  # 单分类器训练集
for i in range(0, y_train.size):
    if y_train[i] == 3:
        class1.append(x_train[i])
class2 = []
class2.append(class1)

print(x_train.shape)

test1 = []  # 单分类器检验集
for i in range(0, y_test.size):
    if y_test[i] == 3:
        test1.append(x_test[i])
test2 = []
test2.append(test1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("加载第两个模型：")
# model = load_model("F://sundries/Models/number.h5")
# middle = load_model("F://sundries/Models/middle.h5")

model = load_model("F://sundries/Models/number_kern.h5")
middle = load_model("F://sundries/Models/middle_kern.h5")

print("构建One Class SVM的测试集")
class3 = []
one_class_test = x_test*0.6     # dim it
re = model.predict(one_class_test)
print("re is like:------------------------------")
print(re)
re2 = np.argmax(re, axis=1)     # 把独热编码变换一下
y_test_10 = np.argmax(y_test, axis=1)
print("re2 is like:-----------------------------")
print(re2)
for i in range(y_test_10.size):
    if re2[i] == 3 and y_test_10[i] != 3:
        class3.append(x_test[i])
class4 = []
class4.append(class3)
print(len(class3))

class5 = []
one_class_test = x_test*0.3     # dim it
re = model.predict(one_class_test)
print("re is like:------------------------------")
print(re)
re2 = np.argmax(re, axis=1)     # 把独热编码变换一下
y_test_10 = np.argmax(y_test, axis=1)
print("re2 is like:-----------------------------")
print(re2)
for i in range(y_test_10.size):
    if re2[i] == 3 and y_test_10[i] != 3:
        class5.append(x_test[i])
class6 = []
class6.append(class5)
print(len(class5))


print("构建SVM训练集")
dense1_output = middle.predict(class2)
print(dense1_output.shape)
print(dense1_output)

print("OneClass 建模")
clf = svm.OneClassSVM(nu=0.4, kernel="rbf")
print("训练OneClass")
clf.fit(dense1_output)

print("测试集过middle层")
SVM_test = middle.predict(class4)
print(SVM_test)

print("测试")
re3 = clf.decision_function(SVM_test)
print(re3)


print("测试集过middle层")
SVM_test = middle.predict(class6)
print(SVM_test)

print("测试")
re3 = clf.decision_function(SVM_test)
print(re3)


print("检验集过middle层")
SVM_test = middle.predict(test2)
print(SVM_test)

print("检验")
re3 = clf.decision_function(SVM_test)
print(re3)

print("验证SVM模型")
re = clf.decision_function(dense1_output)
print(re)