import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 1, 创建一个多元线性回归模型（从一元线性回归->多元线性回归）
class MultipleLinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(MultipleLinearRegression, self).__init__()
        # 创建一个全连接层，全连接层的输入维度是 input_dim，输出的维度是1
        # 这里面的参数 w b，对于自动求导来说，其实已经默认设置了 requires_grad=True
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# 2, 准备数据，自己构造一个数据
# 假设我们有三个特征：x1, x2, x3， y = 2*x1 + 3*x2 +4*x3 + 1 + noise
x_data = torch.tensor(
    [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
    dtype=torch.float32,
)
y_data = torch.tensor(
    [
        [2 * 1.0 + 3 * 2.0 + 4 * 3.0 + 1 + 0.1],
        [2 * 2.0 + 3 * 3.0 + 4 * 4.0 + 1 + 0.2],
        [2 * 3.0 + 3 * 4.0 + 4 * 5.0 + 1 - 0.1],
        [2 * 4.0 + 3 * 5.0 + 4 * 6.0 + 1 - 0.1],
    ],
    dtype=torch.float32,
)

# 3，创建模型、创建损失函数、创建优化器
model = MultipleLinearRegression(
    input_dim=3
)  # 在创建模型的时候，其实里面参数，就会被随机初始化
criterion = nn.MSELoss()  # 回归任务最常用的损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("训练前的参数：")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

# 4, 训练阶段
losses = []
for epoch in range(100):
    # 前向传播
    y_pred = model(x_data)  # wx+b
    loss = criterion(y_pred, y_data)

    # 反向传播
    optimizer.zero_grad()  # 清除梯度
    loss.backward()  # 计算模型参数的梯度
    optimizer.step()  # 更新参数

    losses.append(loss.item())

    # 每隔20次迭代打印一条信息
    if epoch % 20 == 0:
        print(f"Epoch [{epoch}/100], Loss: {loss.item():.4f}")

print("\n训练后的参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")


# 5，测试模型
test_x = torch.tensor([[5.0, 6.0, 7.0]])
prediction = model(test_x)
true_value = 2 * 5.0 + 3 * 6.0 + 4 * 7.0 + 1
print(f"\n预测 x=[5,6,7] 时的 y 值: {prediction.item():.4f}")
print(f"真实值应该是: {true_value}")

# 可视化损失
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
