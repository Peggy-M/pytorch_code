from re import L
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 1.创建一个简单的多元线性回归模型 y = wx + b
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        # 创建一个全连接层的输入输出维度都是 1 )
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# 2.准备数据,自己构造一个数据
# y = 2x + 1 + noise
x_data = torch.tensor(
    [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ]
)
y_data = torch.tensor(
    [
        [3.1],
        [5.2],
        [6.9],
        [8.7],
    ]
)


# 3. 创建模型,创建损失函数,创建优化器
model = SimpleLinearRegression()  # 在创建模型的时候参数就是随机初始化好的 nn.Linear(1,1) 这里就会随机初始化 w 和 b 参数
criterion = nn.MSELoss()  # 回归任务最常使用的损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 告诉对哪一个模型进行优化

print("训练前的参数:")
for param in model.parameters():
    print(f"{param.name}:{param.data}\n")


# 4.训练阶段
losses = []

for epoch in range(100):
    # 前向传播
    # 这里调用相当于是foward方法, 在这里方法的当中的 x 就会执行 , 我们前面定义nn.Linear(1,1) 也就是 y = wx + b
    y_pred = model(x_data)
    # 计算损失
    loss = criterion(y_pred, y_data)  # 预测的结果与真实值的结果

    # 反向传播
    optimizer.zero_grad()  # 清除梯度
    loss.backward()  # 计算模型参数的梯度
    optimizer.step()  # 在不断反向计算的时候更新模型参数

    # 记录损失
    losses.append(loss.item())

    # 每相隔 20次迭代打印一条信息
    if epoch % 20 == 0:
        print(f"epoch:{epoch}/100,loss:{loss.item():.4f}")

print("训练后的参数:")
for param in model.parameters():
    print(f"{param.name} :{param.data}\n")

# 可视化损失值的变化
plt.figure(figsize=(10, 4))  # 创建画布
# 在画布上切分为左右两个子图
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Traning loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")


# 绘制原始的数据点和拟合处理的模型
plt.subplot(1, 2, 2)
plt.scatter(x_data.numpy(), y_pred.detach().numpy(), label="True data")
# 绘制拟合出来的模型,因为没有切换子图的位置
x_plot = torch.linspace(0, 6, 100)
# 这里和 np.linspace 作用一样,都是在 0~6之间创建 100 个点
x_plot = x_plot.reshape(-1, 1)  # 将一维数组转换为二维数组
y_pred = model(x_plot).detach().numpy()
plt.plot(x_plot.numpy(), y_pred, "r-", label="Fitted Line")
plt.legend()  # 添加图例
plt.title("Linear Regression Model")

plt.tight_layout()
plt.show()
