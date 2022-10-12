#%%
# 1.数据读入
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
data1 = pd.read_excel('/data0/BigPlatform/FL/lirongchang/medical/esophagus cancer1/esophagus cancer1/data_new.xlsx', header=None)
# data1.head()
# target = np.array(data1)[1:,5].astype('int')

#%%
import os
import zipfile
import numpy as np
import paddle
# from paddle.nn import functional as F
paddle.__version__
paddle.device.set_device('gpu:6')

feature = np.load('feature.npy').astype(np.float32)

label = np.load('label.npy').astype(np.float32)
#%%

# 在比率70-30中分割数据以进行培训和验证。
x_train = feature[:250]
y_train = label[:250]
x_val = feature[250:]
y_val = label[250:]

print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

#%%
# 按照Dataset的使用规范，构建肺部数据集

from paddle.io import Dataset

class CTDataset(Dataset):
    # 肺部扫描数据集
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, x, y, transform=None):
        """
        步骤二：实现构造函数，定义数据集大小
        Args:
            x: 图像
            y: 图片存储的文件夹路径
            transform (callable, optional): 应用于图像上的数据处理方法
        """
        self.x = x
        self.y = y
        self.transform = transform # 获取 transform 方法

    def __getitem__(self, idx):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据/测试数据，对应的标签）
        """
        img = self.x[idx]
        label = self.y[idx]
        # 如果定义了transform方法，使用transform方法
        if self.transform:
            img,label = self.transform([img,label])
        # 因为上面我们已经把数据集处理好了生成了numpy形式，没必要处理了
        return img, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.y) # 返回数据集大小，即图片的数量

#%%
# 标准化自定义 transform 方法
# 飞桨现在的transform方法只能处理image数据暂时不能处理lable数据，所以我们要定义transform
class TransformAPI(object):
    """
    步骤一：继承 object 类
    """

    def __call__(self, data):
        """
        步骤二：在 __call__ 中定义数据处理方法
        """

        processed_data = data
        return processed_data


import paddle
import random
from scipy import ndimage
import paddle.vision.transforms.functional as F


# 将图像旋转几度
class Rotate(object):

    def __call__(self, data):
        image = data[0]
        key_pts = data[1]
        # 定义一些旋转角度
        angles = [-20, -10, -5, 5, 10, 20]
        # 随机挑选角度
        angle = random.choice(angles)
        # 旋转体积
        image = ndimage.rotate(image, angle, reshape=False)
        image[image < 0] = 0
        image[image > 1] = 1
        return image, key_pts


# 将图像的格式由HWD改为CDHW
class ToCDHW(object):

    def __call__(self, data):
        image = data[0]
        key_pts = data[1]
        image = paddle.transpose(paddle.to_tensor(image), perm=[2, 0, 1])
        image = np.expand_dims(image, axis=0)
        return image, key_pts

from paddle.vision.transforms import Compose

# create the transformed dataset
train_dataset = CTDataset(x_train,y_train,transform=Compose([Rotate(),ToCDHW()]))
valid_dataset = CTDataset(x_train,y_train,transform=Compose([ToCDHW()]))
#%%

class Model3D(paddle.nn.Layer):
    def __init__(self):
        super(Model3D, self).__init__()
        self.layerAll = paddle.nn.Sequential(
            paddle.nn.Conv3D(1, 64, (3, 3, 3)),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool3D(kernel_size=2),
            paddle.nn.BatchNorm3D(64),

            paddle.nn.Conv3D(64, 64, (3, 3, 3)),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool3D(kernel_size=2),
            paddle.nn.BatchNorm3D(64),

            paddle.nn.Conv3D(64, 128, (3, 3, 3)),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool3D(kernel_size=2),
            paddle.nn.BatchNorm3D(128),

            paddle.nn.Conv3D(128, 256, (3, 3, 3)),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool3D(kernel_size=2),
            paddle.nn.BatchNorm3D(256),

            paddle.nn.AdaptiveAvgPool3D(output_size=1),
            paddle.nn.Flatten(),
            paddle.nn.Linear(256, 512),
            paddle.nn.Dropout(p=0.3),

            paddle.nn.Linear(512, 1),
            paddle.nn.Sigmoid()

        )

    def forward(self, inputs):
        x = self.layerAll(inputs)
        return x


model = paddle.Model(Model3D())
model.summary((-1, 1, 64, 128, 128))

#%%

epoch_num = 50
batch_size = 8
batch_size_valid = 10
learning_rate = 0.0001

val_acc_history = []
val_loss_history = []
val_f1_history = []

from sklearn.metrics import f1_score, roc_auc_score

def train(model):
    print('start training ... ')
    # turn into training mode
    model.train()

    # 该接口提供一种学习率按指数函数衰减的策略。
    scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=learning_rate, gamma=0.96, verbose=True)
    opt = paddle.optimizer.Adam(learning_rate=scheduler,
                                parameters=model.parameters())

    train_loader = paddle.io.DataLoader(train_dataset,
                                        shuffle=True,
                                        batch_size=batch_size)

    valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=batch_size_valid)

    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1], dtype="float32")
            y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            bce_loss = paddle.nn.BCELoss()
            loss = bce_loss(logits, y_data)

            if batch_id % 10 == 0:
                print("epoch: {}/{}, batch_id: {}, loss is: {}".format(epoch, epoch_num, batch_id, loss.numpy()))
            loss.backward()
            opt.step()
            opt.clear_grad()

        # evaluate model after one epoch
        model.eval()
        accuracies = []
        losses = []
        list_f1 = []
        list_auc = []
        for batch_id, data in enumerate(valid_loader()):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1], dtype="float32")
            y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            bce_loss = paddle.nn.BCELoss()
            loss = bce_loss(logits, y_data)

            mask = np.float32(logits >= 0.5)  # 以0.5为阈值进行分类
            correct = np.sum(mask == np.float32(y_data))  # 计算正确预测的样本个数
            acc = correct / batch_size_valid  # 计算精度
            accuracies.append(acc)
            # list_auc.append(roc_auc_score(np.float32(y_data), mask))
            list_f1.append(f1_score(np.float32(y_data), mask))
            losses.append(loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        avg_auc, avg_f1 = np.mean(list_auc), np.mean(list_f1)
        print("[validation] epoch: {}/{}, accuracy/loss: {}/{}".format(epoch, epoch_num, avg_acc, avg_loss))
        print("[validation] epoch: {}/{}, auc/f1: {}/{}".format(epoch, epoch_num, avg_auc, avg_f1))
        val_acc_history.append(avg_acc)
        val_loss_history.append(avg_loss)
        val_f1_history.append(avg_f1)
        model.train()
        paddle.save(model.state_dict(), "net_3d.pdparams")



model = Model3D()
train(model)


#%%
# 模型保存
# paddle.save(model.state_dict(), "net_3d.pdparams")

#%%
import matplotlib.pyplot as plt
plt.plot(val_acc_history, label = 'acc')
plt.plot(val_loss_history, label ='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0, 1.1])
plt.legend(loc='lower left')
plt.show()


#%%
import matplotlib.pyplot as plt
plt.plot(val_f1_history, label = 'F1')
plt.plot(val_loss_history, label ='loss')
plt.xlabel('Epoch')
plt.ylabel('F1/Loss')
plt.ylim([0, 1.1])
plt.legend(loc='lower left')
plt.show()
