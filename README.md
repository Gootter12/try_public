#### 文件说明

middle.py 用来生成模型，包括训练好的分类器模型和其中的一个中间层

dim.py 用于取用middle.py训练好的模型并训练OneClass SVM，其对测试集的处理为\*0.6或\*0.3使其变暗

~~Models 用于存放训练好的模型，_kern表示分类器包含卷积层~~

picture edit/ 目前有四个用于处理图片的操作