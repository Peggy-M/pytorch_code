import os
import numpy as np
from cProfile import label
from torchvision import transforms,datasets

# 创建保存目录
save_dir = "./mnist_numpy"
os.makedirs(save_dir, exist_ok=True)

# 获取 MNIST 数据集
transforms = transforms.Compose([transforms.ToTensor()]) # 将图片转换为张量
train_data = datasets.MNIST(root="./data",train=True,download=True,transform=transforms)
test_data = datasets.MNIST(root="./data",train= False,download=True,transform=transforms)

# 提取训练数据
train_images = []
train_labels = []
for i in range(len(train_data)):
    image,lable =train_data[i]
    # sueeze() 会将 (H,W,1)->(H,W) (1,H,W)->(H,W) 也就是将三维转为二维的
    # 通过 numpy 将原来的张量转为 numpy 数组, astype 将数组里面的数据类型转为 uint8 的这种数据类型
    image = (image.squeeze() * 255).numpy().astype(np.uint8)
    train_images.append(image)
    train_labels.append(lable)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# 提取测试数据
test_images = []
test_labels = []
for i in range(len(test_data)):
    image,lable =test_data[i]
    # sueeze() 会将 (H,W,1)->(H,W) (1,H,W)->(H,W) 也就是将三维转为二维的
    # 通过 numpy 将原来的张量转为 numpy 数组, astype 将数组里面的数据类型转为 uint8 的这种数据类型
    image = (image.squeeze() * 255).numpy().astype(np.uint8)
    test_images.append(image)
    test_labels.append(lable)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# 保存成为 numpy 文件
np.save(os.path.join(save_dir,'train_images.npy'),train_images)
np.save(os.path.join(save_dir,'train_labels.npy'),train_labels)
np.save(os.path.join(save_dir,'test_images.npy'),test_images)
np.save(os.path.join(save_dir,'test_labels.npy'),test_labels)

print(f"数据保存成功,已经保存到:{save_dir}")