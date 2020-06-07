import torch, torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

#超参数设置
# device = torch.device('GPU' if torch.cuda.is_available() else 'CPU')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 80
batch_size = 100
learning_rate = 0.001


#准备数据
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
])

train_set = datasets.CIFAR10('CIFAR10', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = datasets.CIFAR10('CIFAR10', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


#搭建模型
def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_class=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = torch.nn.AvgPool2d(8)
        self.fc = torch.nn.Linear(64, num_class)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = torch.nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride), torch.nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model = ResNet(ResidualBlock, [2,2,2])
#构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups():
        param_group['lr'] = lr



#训练阶段
total1 = len(train_loader)
curr_lr = learning_rate
train_loss = 0
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):


        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        optimizer.step()
        train_loss += loss.item()

        if (i+1) % 100 == 0:
            print('epoch:[{}/{}], iteration[{}/{}, loss:{}]'.format(epoch+1, num_epochs, i+1, total1, train_loss / 100))
            total_loss = 0

    if (epoch+1) % 20 ==0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)



#预测阶段
model.eval()
correct = 0
total2 = len(test_loader)
for inputs, targets in test_loader:


    outputs = model(inputs)
    _, preds = torch.max(outputs, dim=1)
    correct += (preds == targets).sum().item()

    print('acc on test: %', 100* correct / total2)

torch.save(model.state_dict(), 'residual_nn.ckpt')