import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

#超参数
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
num_epochs = 5
num_classes = 10
learning_rate = 0.001
batch_size = 100


#准备数据
train_set = torchvision.datasets.MNIST('MNIST', train=True, transform=transforms.Compose([transforms.ToTensor(),]))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.MNIST('MNIST', train=False, transform=transforms.Compose([transforms.ToTensor(),]))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

#搭建模型
class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(32*7*7, num_classes)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(batch_size, -1)
        return self.fc(x)

model = ConvNet(num_classes=num_classes).to(device)
#构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)


#训练阶段
#len是计算第0个维度的长度
total = len(train_loader)
batch_loss = 0
model.train()
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
        if (i+1) % 100 == 0:
            print('epoch:[{}/{}], iteration[{}/{}, loss:{:.4f}]'.format(epoch+1, num_epochs, i+1, total, batch_loss/100))
            #重新计算下100batch的平均损失
            batch_loss = 0

#测试阶段
correct = 0
batch = 0
model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == targets).sum().item()
        batch += outputs.size(0)
        # total1 += len(outputs) #全部batch
    print('acc on test: {:.4f}%'.format(correct / batch * 100))


# torch.save(model.state_dict(), 'convolution_nn.ckpt')
#
# linear_model = model.load_state_dict(torch.load('convolution_nn.ckpt'))
#
# #查看保存模型的参数
# for k, v in torch.load('convolution_nn.ckpt').items():
#     print(k,v)