import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#超参数
input_size = 28 * 28
num_classes = 10 #output_size
learning_rate = 0.001
num_epochs = 5
batch_size = 100

#准备数据
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_set = torchvision.datasets.MNIST('MNIST', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.MNIST('MNIST', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# #搭建模型
# class LogisticRgression(nn.Module):
#     def __init__(self):
#         super(LogisticRgression, self).__init__()
#         self.linear = nn.Linear(768, 10)
#
#     def forward(self, x):
#         x = x.view(-1)
#         x = self.linear(x)
#         return x

model = nn.Linear(input_size, num_classes)

#构建损失函数和优化器
criterion = nn.CrossEntropyLoss()#自动计算softmax=logsumexp
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)


#训练阶段
total_step = len(train_loader)
total_loss = 0
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.view(-1, input_size)
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算100个batch的平均损失
        total_loss += loss.item()
        if (i+1) % 100 == 0:
            print('epoch:[{}/{}], iteration[{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()/100))
            total_loss = 0

        #计算第100个batch的损失
        # if (i+1) % 100 == 0:
        #     print('epoch:[{}/{}], iteration[{}/{}], loss:{%.4f}'.format(epoch+1, num_epochs, i+1, total_step), loss.item())


        #计算整个数据集的平均损失
        # total_loss += loss.item() * len(inputs.size(0))
        #print('loss:{%.4f}'.format(loss.item()/total_step)


#预测阶段
correct = 0
total = 0
with torch.no_grad():
    for inputs, target in test_loader:
        inputs = inputs.view(-1, input_size)
        outputs = model(inputs)
        _, pred = torch.max(outputs, dim=1)#在num_class维度上取最大值
        correct += (pred == targets).sum().item()
        total += len(pred)

        print(len(outputs), len(pred), inputs.size()[0])
    print('acc on test:{}%'.format(correct / total * 100))


torch.save(model.state_dict(), 'logistic_regression.ckpt')

logistic_model = model.load_state_dict(torch.load('logistic_regression.ckpt'))

#查看保存模型的参数
for k, v in torch.load('logistic_regression.ckpt').items():
    print(k,v)