# PyTorch 学习实战项目 (PyTorch Learning Project)

欢迎来到我的 PyTorch 学习项目！本项目主要以经典的 **MNIST 手写数字识别** 为核心切入点，从最基础的深度学习模型搭建，一路进阶到数据本地化处理、自定义数据集加载、模型可视化监控以及混合精度训练。

通过这个项目，我系统性地掌握了使用 PyTorch 进行深度学习开发的完整工程流。

---

## 📚 知识点大纲 (Knowledge Outline)

### 1. PyTorch 基础知识 (PyTorch Basics)
*   **自动求导机制 (Autograd)**: 
    *   理解 `requires_grad=True` 张量属性，学习自动生成计算图。
    *   掌握 `backward()` 标量与向量反向传播求导。
    *   理解并应用梯度累积 (Gradient Accumulation) 以及梯度清零 (`grad.zero_()`)。
    *   熟悉 `torch.no_grad()` 和 `.detach()` 以剥离计算图（节省内存及推理时使用）。
*   **简单线性回归搭建**: 通过一个一元一次的方程 `y = wx + b` 掌握最原始的神经网络结构 (`nn.Linear(1, 1)`) 与前向传播基础。

### 2. 深度学习模型构建 (Model Construction)
*   **卷积神经网络 (CNN) 设计**: 
    *   使用 `torch.nn.Conv2d` 提取图像特征。
    *   使用 `torch.nn.MaxPool2d` 进行下采样降维。
    *   使用 `torch.nn.Linear` 搭建全连接层输出分类结果。
*   **防过拟合技巧**: 引入 `nn.Dropout` 在训练时随机丢弃神经元。
*   **前向传播逻辑**: 熟悉 `forward` 函数的编写，以及特征图在网络中的维度变换（如使用 `view` 或 `reshape` 将 4D 张量展平为 2D）。

### 3. 数据处理与加载 (Data Processing & Loading)
*   **官方数据集调用**: 使用 `torchvision.datasets.MNIST` 快速下载和加载标准数据。
*   **数据预处理流水线**: 使用 `transforms.Compose` 结合 `ToTensor()` 与 `Normalize()` 进行归一化，提升模型收敛稳定性。
*   **自定义数据集开发 (重点)**: 
    *   继承 `torch.utils.data.Dataset`。
    *   实现 `__len__` 和 `__getitem__` 魔法方法。
    *   掌握了从 `.npy` (Numpy 数组) 读取数据的 `CustomMNISTDataset` 实现。
    *   掌握了基于 Pandas 读取 `.csv` 标签并结合 `PIL.Image` 读取本地 `.png` 图像文件的 `CustomMNISTImageDataset` 实现。
*   **数据加载器**: 使用 `DataLoader` 实现数据的批处理 (Batching) 和打乱 (Shuffle)。

### 4. 模型训练与评估 (Training & Evaluation)
*   **训练循环 (Training Loop)**: 
    *   设备迁移: 使用 `.to(device)` 实现 CPU 与 GPU (`cuda`) 的无缝切换。
    *   前向传播 -> 计算损失 (`nn.CrossEntropyLoss`) -> 梯度清零 (`optimizer.zero_grad()`) -> 反向传播 (`loss.backward()`) -> 参数更新 (`optimizer.step()`)。
*   **优化器选择**: 使用 `torch.optim.Adam` 作为模型参数的优化器。
*   **评估指标**: 计算并打印每个 Epoch 的平均 Loss 与模型分类准确率 (Accuracy)。

### 5. 训练可视化与结果展示 (Visualization)
*   **Matplotlib 绘图**:
    *   记录训练过程中的 Train Loss 和 Test Loss，并绘制折线图。
    *   配置中文字体 (如 `SimHei`) 以正常显示图表标题和坐标轴。
*   **预测结果可视化**: 从测试集中抽取样本输入模型，将图像显示出来的同时，对比 **真实标签** 与 **模型预测标签**。

### 6. 高级特性与工程化 (Advanced Features)
*   **数据本地化落盘**: 编写脚本将原本封装在 PyTorch 内部的数据提取出来，保存为原生的 `.npy` 数组文件或 `.png` 图片+`.csv` 标签文件。
*   **模型持久化**: 使用 `torch.save(model.state_dict(), 'xxx.pth')` 保存训练好的模型权重。
*   **TensorBoard 监控**: 
    *   引入 `SummaryWriter`。
    *   在训练过程中实时记录 Loss、Accuracy 和学习率 (`Learning Rate`) 的变化，通过 Web 界面直观监控训练走势。
*   **混合精度训练 (AMP)**: 
    *   引入 `torch.cuda.amp.autocast` 和 `GradScaler`。
    *   通过 `scaler.scale(loss).backward()` 等操作，在不损失模型精度的情况下，降低显存占用并大幅提升 GPU 训练速度。

---

## 📂 核心代码文件导航 (File Structure)

*   **`PyTorch 自动求导.ipynb`**: 详细演示了 PyTorch 的底层核心：张量的求导、计算图、`detach` 与 `no_grad()` 的用法。
*   **`PyTorch简单线性回归.py`**: 用极简的线性回归演示 `nn.Module` 与 `nn.Linear` 搭建。
*   **`实现 CNN 预测 MNIST.py`**: 项目的主入口之一，包含了标准的 CNN 定义、训练、评估、绘图和预测全流程。
*   **`落地MNIST到本地磁盘.py`**: 将 MNIST 数据集转换为 Numpy (`.npy`) 文件并保存到本地。
*   **`MNISTImageSaver.py`**: 将 MNIST 数据集按类别保存为本地 `.png` 图片，并生成配套的 `labels.csv` 文件。
*   **`CustomMNISTDataset.py`**: 自定义 Dataset 类，提供了从 `.npy` 和图片文件中读取数据的两种实现。
*   **`Local_Data_MNIST.py` / `LoaclMNISTLoaderImages.py`**: 结合自定义数据集进行模型训练的实战脚本。
*   **`实现CNN预测 MNIST_Tensorboard.py`**: 加入了 TensorBoard 监控功能的训练脚本。
*   **`混合精度训练.py`**: 展示了如何使用 PyTorch AMP 进行高效混合精度训练。

---
*记录于我的 PyTorch 学习之旅。*
