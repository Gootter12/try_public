from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.models import Model
from sklearn import svm
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




y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()

model.add(Convolution2D(64, kernel_size=(3, 3), border_mode='same', input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size=(3, 3), border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, name='Dense_1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, name='Dense_2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=512, epochs=4)

loss, accuracy = model.evaluate(x_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)

print("取中间网络")
dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('Dense_1').output)
print("输入")
dense1_output = dense1_layer_model.predict(class2)
print(dense1_output.shape)

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

print("构建测试集")
testSVM = dense1_layer_model.predict(class4)

print("OneClass 建模")
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
print("训练OneClass")
clf.fit(dense1_output)

print("测试")
re = clf.decision_function(testSVM)
print(re)

print("保存模型")
dr1 = "F://sundries/Models/number_kern.h5"
model.save(dr1)
dr2 = "F://sundries/Models/middle_kern.h5"
dense1_layer_model.save(dr2)