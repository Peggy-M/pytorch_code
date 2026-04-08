import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from CustomMNISTDataset import CustomMNISTDataset


# 1.定义算法 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层: 输入的通道是 1, 输出的是 32,卷积核 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 第二个卷积层: 输入的通道是 32, 输出的是 64,卷积核 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # dropout 层 可选项,用于防止过拟合,本质是训练时使用一部分参数,预测时使用所有参数
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 28*28->14*14->7*7 64个7x7的特征图
        self.fc2 = nn.Linear(128, 10)  # 输出 10 个类别

    def forward(self, x):
        # 第一个卷积块: 卷积-> ReLU -> 池化 -> Dropout
        x = self.pool(F.relu(self.conv1(x)))
        # 第二个卷积块: 卷积-> ReLU -> 池化 -> Dropout
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)

        # Flatten 4D -> 2D,python 来说其实就是 reshape
        x = x.view(-1, 64 * 7 * 7)
        # 全连接
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# 2.数据的预处理
# compose 将一些数据与处理的逻辑组合在一起
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 将数据转为 Tensor
        transforms.Normalize(
            (0.1307,), (0.3081,)
        ),  # 数据进行归一化,防止数据过拟合,减少噪声
    ]
)

# 3.读取本地的 MNIST 的 npy 文件
train_datasets = CustomMNISTDataset(
    data_dir="./mnist_numpy", train=True, transfrom=transform
)

test_datasets = CustomMNISTDataset(
    data_dir="./mnist_numpy", train=False, transfrom=transform
)


# 3.加载数据集 (transform=transform 加载的时候并进行预处理)
# train_datasets = datasets.MNIST(
#     root="./data", train=True, download=True, transform=transform
# )
# test_datasets = datasets.MNIST(
#     root="./data", train=False, download=True, transform=transform
# )

# 5.创建数据加载器 (每一次迭代从数据集加载 64 个样本,进行正向传播-反向传播,如果是训练集,shuffle=True则表示打乱数据)
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
# 测试集只做正向传播,所以这里的加载批量大小可以大一些
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1000, shuffle=False)


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备:{device}")

# 6.初始化模型、损失函数、优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数,用于分类任务
# 优化器: Adam, 学习率: 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 新增混合精度训练所需要的 GradScaler
scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None


# 6, 训练函数逻辑
def train(model, device, train_loader, optimizer, epoch, scaler):
    """
    model: 针对哪个模型进行训练
    device: 基于CPU运算还是基于GPU运算
    train_loader: 通过它获得训练数据
    optimizer: 通过它对模型参数进行update更新
    epoch: 当前训练第几轮了，训练都是分轮次分批次进行训练
    """
    model.train()  # 将模型设置为训练模式
    train_loss = 0
    correct = 0  # 记录训练过程中模型的正确率
    # 接下来就分批次进行训练
    for batch_idx, (data, target) in enumerate(train_loader):
        # data 是一个批次的数据 X，也就是一个批次64张图片
        # target 是一个批次的数据 Y，也就是一个批次64张图片对应的分类号0~9
        # 需要把数据和模型仍到同一个设备里面去
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度清零

        # 使用 autocast 进行混合精度前向传播
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            output = model(data)  # 正向传播
            loss = criterion(output, target)  # 计算损失

        if scaler is not None:
            # 反向传播，本质就是会用loss对每个参数求偏导(梯度)
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # 参数更新，也就是应用相对应的公式
            scaler.update()  # 更新缩放因子
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()  # train_loss 相当在把每个批次的loss进行加和
        # dim=1 的意思是说按照索引1这个维度去求最大
        # 获取预测结果，得到的就是分值最大的类别号
        prediction = output.argmax(dim=1, keepdim=True)
        # target.view_as(prediction) 相当于是把target整成和prediction形状一样的张量
        # 一个批次中判断正确的样本数量，累加到correct变量中
        correct += prediction.eq(target.view_as(prediction)).sum().item()

        # 对100求余看看是否等于0；每隔100次打印一下信息
        if batch_idx % 100 == 0:
            print(
                f"训练 Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}"
            )

    train_loss /= len(
        train_loader
    )  # 平均批次损失，累积的批次损失除以一轮中总的批次数量
    # 正确率，累积的模型预测正确的样本数除以总的样本数
    accuracy = 100.0 * correct / len(train_loader.dataset)
    print(
        f"训练集: 平均损失：{train_loss:.4f}, 准确率: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%:)"
    )
    return train_loss, accuracy


# 7.测试函数逻辑
def test(model, device, test_loader, criterion):
    """
    model: 针对哪个模型进行测试
    device: 基于CPU运算还是基于GPU运算
    test_loader: 通过它获得测试数据
    criterion: 通过它对模型的输出进行损失计算
    """
    model.eval()  # 将模型设置为评估模式
    test_loss = 0
    correct = 0  # 记录测试过程中模型的正确率

    with torch.no_grad():  # 下面的操作不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # dim=1 的意思是说按照索引1这个维度去求最大
            prediction = output.argmax(
                dim=1, keepdim=True
            )  # 获取预测结果，得到的就是分值最大的类别号
            # target.view_as(prediction) 相当于是把target整成和prediction形状一样的张量
            # 一个批次中判断正确的样本数量，累加到correct变量中
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"测试集: 平均损失：{test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%):"
    )
    return test_loss, accuracy


# 8.开始训练
epochs = 5
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = train(
        model, device, train_loader, optimizer, epoch, scaler
    )
    test_loss, test_accuracy = test(model, device, test_loader, criterion)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# 9.保存模型
torch.save(model.state_dict(), "mnist_cnn.pth")
print("模型已保存 mnist_cnn.pth")


# 11, 预测示例函数
def predict_and_show(model, test_loader, device, num_images=10):
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 显示图片和预测结果
    fig = plt.figure(figsize=(15, 3))
    for i in range(num_images):
        ax = fig.add_subplot(1, num_images, i + 1)
        img = images[i].cpu().numpy().squeeze()
        ax.imshow(
            img, cmap="gray"
        )  # 手写数字图片是黑白图片，所以这里需要设置 cmap='gray'
        ax.set_title(f"真实: {labels[i].item()}\n预测: {predicted[i].item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# 12，显示预测结果
predict_and_show(model, test_loader, device)
