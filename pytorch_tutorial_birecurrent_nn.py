import torch, torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader


#超参数 设置
num_epochs = 5
num_class = 10
learning_rate = 0.001
batch_size = 100
hidden_size = 128
num_layers = 2
seq_len = 28
input_size = 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#准备数据
train_set = torchvision.datasets.MNIST('MNIST', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

test_set = torchvision.datasets.MNIST('MNIST', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size)

#搭建模型
class BiRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #batch_first只对输入输出有效，不影响隐藏层维度
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.fc = torch.nn.Linear(2*hidden_size, num_class)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        #h0和c0可写不写，反正都是初始化为0作为输入 hidden的维度是num_layers * num_direction, batch_size, hidden_size*2
        #rnn和gru没有cell
        # out, hidden = self.lstm(x, (h0, c0))
        # hidden = out[:, -1, :]

        #注意这样写hidden就会是二维的 batch_size * hidden_size
        out, (hidden, cell) = self.lstm(x,(h0, c0))
        hidden = torch.cat((hidden[-1], hidden[-2]), dim=1)
        x = self.fc(hidden)
        return x

model = BiRNN(input_size, hidden_size, num_layers, num_class).to(device)


#构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#训练阶段
train_length = len(train_loader)
train_loss = 0
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device) #batch_size, 1, seq_len, input_size
        targets = targets.to(device)
        inputs = inputs.view(-1, seq_len, input_size)
        # inputs = inputs.permute(1, 0, 2)
        # inputs = inputs.view(seq_len, -1, input_size)

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if (i+1) % 100 == 0:
            print('epoch:[{}/{}], iteration[{}/{}], loss:{}'.format(epoch+1, num_epochs, i+1, train_length, train_loss / 100))
            train_loss = 0


#预测阶段
correct = 0
test_length = len(test_loader)
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs = inputs.view(-1, seq_len, input_size)
        outputs = model(inputs)

        _, preds = torch.max(outputs, dim=1)
        correct += (targets == preds).sum().item()

    print('acc on test:{:.4f}%'.format(correct / test_length))



#保存模型参数
# torch.save(model.state_dict(), 'birecurrent_nn.ckpt')

