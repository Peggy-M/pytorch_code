import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Any
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter



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

# 3.加载数据集 (transform=transform 加载的时候并进行预处理)
train_datasets = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_datasets = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

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

# 创建 SummaryWriter ,指定把数据写入到那个目录下,保存日志在 runs/mnist_cinn
log_dir = "runs/mnist_cinn" # 日志记录
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir) # 创建 SummaryWriter 实例对象

# 可选项: 将模型结构写入到 Tensorboard
# 输入数据的形状保持一致(batch_size,channel,height,weight)
dummy_input = torch.randn(1,1,28,28) 
writer.add_graph(model, dummy_input) # 将模型结构写入到 Tensorboard 中





# 6, 训练函数逻辑
def train(model, device, train_loader, optimizer, epoch):
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
        output = model(data)  # 正向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播，本质就是会用loss对每个参数求偏导(梯度)
        optimizer.step()  # 参数更新，也就是应用相对应的公式

        train_loss += loss.item()  # train_loss 相当在把每个批次的loss进行加和
        # dim=1 的意思是说按照索引1这个维度去求最大
        prediction = output.argmax(
            dim=1, keepdim=True
        )  # 获取预测结果，得到的就是分值最大的类别号
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
    # 这些是标量
    # 记录训练损失和准确率到 Tensorboard
    writer.add_scalar("Training Loss", train_loss, epoch)
    writer.add_scalar("Training Accuracy", accuracy, epoch)
    # optimizer.param_groups[0]["lr"] 是当前轮次的学习率 0 表示第一个参数组的学习率
    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
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
    # 这些是标量
    # 记录测试损失和准确率到 Tensorboard
    writer.add_scalar("Test Loss", test_loss, epoch)
    writer.add_scalar("Test Accuracy", accuracy, epoch)
    # optimizer.param_groups[0]["lr"] 是当前轮次的学习率 0 表示第一个参数组的学习率
    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
    return test_loss, accuracy


# 8.开始训练
epochs = 5
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
    test_loss, test_accuracy = test(model, device, test_loader, criterion)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)


# 写入各类别的准确率
# 13，计算每个类别的准确率
def class_accuracy(model, device, test_loader,epoch,writer):
    model.eval()
    class_correct = list[float](0. for _ in range(10))
    class_total = list[float](0. for _ in range(10))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == target).squeeze()  # 里面存放的是True或者False
            # target.size(0) 这个对应的是样本数量
            for i in range(target.size(0)):
                label = target[i] # 对应图片的标签，分类号
                class_correct[label] += c[i].item() # 在对应类别上面累加1或0
                class_total[label] += 1  # 相当于是counter计数器，把每个类别对应的预测的样本数量进行累加

        class_acc_text ="类别准确率:\n"        
        for i in range(10):
            if class_total[i] > 0:
               acc = 100 * class_correct[i]/class_total[i]
               class_acc_text+=f"数字 {i}: 准确率: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})"
            else:
                class_acc_text+=f"数字 {i}: 没有样本"

            # 写入 tensorboard
            writer.add_text("类的准确率:",class_acc_text,epoch)
            print("\n"+class_acc_text)

print("\n各类别准确率:")
class_accuracy(model, device, test_loader,epoch,writer)

# 9.保存模型
torch.save(model.state_dict(), "mnist_cnn.pth")
print("模型已保存 mnist_cnn.pth")


# 遍历模型参数
for name, param in model.named_parameters():
    writer.add_histogram(name,param, epoch) # 将参数绘制到直方图中
    if param.grad is not None: # 如果存在梯度,将梯度拿出来
        writer.add_histogram(f'{name}.grad', param.grad, epoch) # 将梯度绘制到直方图中

# 记录预测图像 image
def log_predictions(model,device,test_loader,writer,epoch,num_images =10):
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 将图像从 [-1,1] 转回 [0,1]以方便显示
    images_grid = images[:num_images]
    images_denorm = images_grid * 0.3081 + 0.1307 # 反归一化
    grid  = torch.cat([img for img in images_denorm],dim = 2)
    grid = grid.cpu().numpy().squeeze()

    # 写入到 tensorboard
    writer.add_image("Predictions",torch.tensor(grid).unsqueeze(0),epoch)

    # 同时还要给图像打印真实的标签和预测的标签
    pred_text = " | ".join([f"真:{l.item()}, 预:{p.item()}" for l, p in 
                            zip(labels[:num_images], predicted[:num_images])])

    writer.add_text("Predictions",pred_text,epoch)
    
   
log_predictions(model,device,test_loader,writer,epoch)

# 关闭 tensorboard
writer.close()

# SummaryWriter
# add_scaler 添加了损失、准确率和学习率，tensorboard就可以看到它们的曲线
# add_graph  添加了模型结构图
# add_image  添加了预测图像，可在tensorboard进行展示
# add_text   添加了各个类别准确率文本信息
# add_historgram 添加了权重和梯度的分布图
# 现在咱们的代码不仅功能完整的，还具备了工业级训练监控能力！