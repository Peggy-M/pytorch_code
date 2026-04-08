import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import pandas as pd


# 自定义 Dataset 类
class CustomMNISTDataset(Dataset):
    def __init__(self, data_dir, train=True, transfrom=None):
        """
        初始化自定义数据集
        :param data_dir: 数据集目录
        :param train: 是否为训练集
        :param transfrom: 数据转换
        """
        self.data_dir = data_dir
        self.train = train
        self.transfrom = transfrom

        # 根据 train 参数确定数据文件路径
        if train:
            self.images_file = os.path.join(self.data_dir, "train_images.npy")
            self.lables_file = os.path.join(self.data_dir, "train_labels.npy")
        else:
            self.images_file = os.path.join(self.data_dir, "test_images.npy")
            self.lables_file = os.path.join(self.data_dir, "test_labels.npy")

        # 加载数据
        self.images = np.load(self.images_file, allow_pickle=True)
        self.lables = np.load(self.lables_file, allow_pickle=True)

    def __len__(self):
        # 返回数据集的样本数量(有多少个图片就有多少个样本数据)
        return len(self.images)

    def __getitem__(self, index):
        """
        对读取的数据集进行一些预处理转化
        """
        image = self.images[index]  # 根据索引获取图片
        label = self.lables[index]  # 根据索引获取标签

        # 使用 Pillow 库,将一个numpy库数组转为一个灰度的图像对象
        # image.astype('uint8') 将数组 image 数据类型转为 8位 无符号整数 (取值范围转为 0~255)
        image = Image.fromarray(image.astype("uint8"), mode="L")

        # 应用转化函数
        if self.transfrom:
            image = self.transfrom(image)

        return image, label


class CustomMNISTImageDataset(Dataset):
    def __init__(self, data_dir, train=True, transfrom=None):
        """
        初始化自定义数据集
        :param data_dir: 数据集目录
        :param train: 是否为训练集
        :param transfrom: 数据转换
        """
        self.data_dir = data_dir
        self.train = train
        self.transfrom = transfrom

        # 确定数据目录和标签文件
        if train:
            self.data_dir = os.path.join(data_dir, "train")
            self.labels_file = os.path.join(data_dir, "train_lables.csv")
        else:
            self.data_dir = os.path.join(data_dir, "test")
            self.labels_file = os.path.join(data_dir, "test_lables.csv")

        # 读取标签文件
        self.labels_df = pd.read_csv(self.labels_file)

    def __len__(self):
        return len(self.labels_df)  # 统计标签的数量，标签的数量就是图片数据集的数量

    def __getitem__(self, index):
        # 用户确保 idx 是 python 的原生数据类型,比如整数或列表
        if torch.is_tensor(index):
            index = index.tolist()

        # 获取文件列表和标签
        row = self.labels_df.iloc[index]
        filename = row["filename"]
        label = row["label"]

        # 构建完整路径并加载图像
        img_path = os.path.join(self.data_dir, filename)
        image = Image.open(img_path).convert("L")  # 转为灰度图

        # 应用转换
        if self.transfrom:
            image = self.transfrom(image)

        return image, label
