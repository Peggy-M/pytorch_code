from typing import Any

import pandas as pd
from cProfile import label
import numpy as np
import os
from sympy import true
import torch
from torchvision import datasets, transforms
from PIL import Image


# 创建目录结构
def crate_directory_struture():
    base_dir = "./mist_images"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "train"), exist_ok=true)
    os.makedirs(os.path.join(base_dir, "test"), exist_ok=True)

    # 为每一个类型创建子目录
    for i in range(10):
        os.makedirs(os.path.join(base_dir, "train", str(i)), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "test", str(i)), exist_ok=True)

    return base_dir


# 保存数据为图片文件
def save_mnist_as_images():
    base_dir = crate_directory_struture()

    # 定义 transform 并且加载自带的数据集
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=False, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=False, transform=transform
    )

    # 保存训练数据
    print("保存训练数据...")
    train_labels = []
    for i, (image, label) in enumerate[Any](train_dataset):
        # 转化为 PIL 图像
        image_pil = transforms.ToPILImage()(image)
        # 保存图像到对应类别的文件夹
        image_path = os.path.join(base_dir, "train", str(label), f"{i:05d}.png")
        image_pil.save(image_path)
        # 记录文件名和标签
        train_labels.append([f"{label}/{i:05d}.png", label])

        if (i + 1) % 10000 == 0:
            print(f"已保存{i + 1}张训练图像")

    # 保存测试数据
    print("保存测试数据...")
    test_labels = []
    for i, (image, label) in enumerate[Any](test_dataset):
        # 转化为 PIL 图像
        image_pil = transforms.ToPILImage()(image)
        # 保存图像到对应类别的文件夹
        image_path = os.path.join(base_dir, "test", str(label), f"{i:05d}.png")
        image_pil.save(image_path)
        # 记录文件名和标签
        test_labels.append([f"{label}/{i:05d}.png", label])

        if (i + 1) % 10000 == 0:
            print(f"已保存{i + 1}张训练图像")

    # 保存标签信息到本地
    train_labels_df = pd.DataFrame(train_labels, columns=["filename", "label"])
    test_labels_df = pd.DataFrame(test_labels, columns=["filename", "label"])
    train_labels_df.to_csv(os.path.join(base_dir, "train_lables.csv"), index=False)
    test_labels_df.to_csv(os.path.join(base_dir, "test_lables.csv"), index=False)

    print(f"数据保存到:{base_dir}")
    print(f"训练样本数:{len(train_labels)}")
    print(f"测试样本数:{len(test_labels)}")


save_mnist_as_images()
