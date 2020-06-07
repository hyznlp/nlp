import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#超参数设置
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

#准备数据
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [7.09],
                   [8.91], [9.44], [9.98], [10.93], [13.44], [14.20],
                   [14.88], [17.98], [19.01]], dtype=np.float32)

y_train = np.array([[1.3], [2.4], [3.5], [3.71], [4.93], [5.09],
                   [6.91], [7.44], [7.98], [7.93], [9.44], [11.20],
                   [12.8], [12.9], [15.01]], dtype=np.float32)

#搭建模型
model = nn.Linear(input_size, output_size)

#构建损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

#训练阶段
for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    outputs = model(inputs)

    loss = criterion(targets, outputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))



# predicted = model(torch.from_numpy(x_train)).detach().numpy()
# plt.plot(x_train, y_train, 'ro', label='original data')
# plt.plot(x_train, predicted, label = 'fitting data')
# plt.legend()
# plt.show()

#查看模型参数
# print(model.state_dict()['weight'])
print(model.weight, model.bias)

torch.save(model.state_dict(), 'linear_regression.ckpt')
linear_model = model.load_state_dict(torch.load('linear_regression.ckpt'))

#查看保存模型的参数
for k, v in torch.load('linear_regression.ckpt').items():
    print(k,v)