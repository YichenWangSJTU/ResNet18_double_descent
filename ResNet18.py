import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import time
# from drive.MyDrive.Colab.ResNet18.model import make_resnet18k
from model import make_resnet18k
import matplotlib.pyplot as plt
import json
import numpy as np


# CIFAR-10 数据加载并预加载到内存
def load_cifar10(batch_size, noise_ratio=0.2, num_classes=10):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # 将数据加载到内存
    train_data = torch.stack([data[0] for data in train_dataset])
    train_targets = torch.tensor([data[1] for data in train_dataset])
    test_data = torch.stack([data[0] for data in test_dataset])
    test_targets = torch.tensor([data[1] for data in test_dataset])

    # 添加噪声到训练标签
    num_samples = len(train_targets)
    num_noisy = int(noise_ratio * num_samples)
    noisy_indices = torch.randperm(num_samples)[:num_noisy]  # 随机选择10%的样本

    # 替换标签为随机类别（不等于原始类别）
    for idx in noisy_indices:
        original_label = train_targets[idx].item()
        noisy_label = original_label
        while noisy_label == original_label:
            noisy_label = np.random.randint(0, num_classes)
        train_targets[idx] = noisy_label

    # 使用 TensorDataset 包装数据并创建 DataLoader
    train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_targets), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 训练模型
def train_model(model, criterion, optimizer, train_loader, device, epochs):
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * target.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

        train_losses.append(total_loss / total)
        train_accuracies.append(100. * correct / total)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.2f}%")

    return train_losses, train_accuracies

# 计算误差
def calculate_error(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

    return 1 - correct / total

# 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 绘制训练过程中的损失和准确率
def plot_training_curves(losses, accuracies):
    epochs = range(1, len(losses) + 1)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epochs, losses, label="Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)", color="tab:orange")
    ax2.plot(epochs, accuracies, label="Accuracy", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    plt.title("Training Loss and Accuracy")
    plt.show()


# 实验设置
batch_size = 128
epochs = 200
#widths = [1, 2, 4, 6, 8, 9, 10, 12, 16, 32, 64]  # 不同的网络宽度 k
widths = [6, 8, 9, 10, 12]  # 不同的网络宽度 k
results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = load_cifar10(batch_size)

# 开始实验
for k in widths:
    print(f"Training ResNet18 with width k={k}")

    model = make_resnet18k(k=k, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0)

    start_time = time.time()
    train_losses, train_accuracies = train_model(model, criterion, optimizer, train_loader, device, epochs)
    end_time = time.time()
    print("training time:", end_time - start_time)

    train_error = calculate_error(model, train_loader, device)
    test_error = calculate_error(model, test_loader, device)
    num_params = count_parameters(model)

    print(f"Width k={k}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}, Params: {num_params}")

    results.append({
        "width": k,
        "train_error": train_error,
        "test_error": test_error,
        "num_parameters": num_params
    })

    # 绘制当前模型的训练曲线
    plot_training_curves(train_losses, train_accuracies)

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

# 输出结果
for result in results:
    print(result)
