import torch, torchvision, torchtext
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

#超参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
batch_size = 32
embedding_size = 100
hidden_size = 100
max_vocab_size = 50000
learning_rate = 0.001
num_epochs = 2
grad_clip = 5.0

#准备数据
text = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path='',
                                                                     train='text8.train.txt',
                                                                     validation='text8.dev.txt',
                                                                     test='text8.test.txt', text_field=text)

train_loader, val_loader, test_loader = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=batch_size, device=device, bptt_len=50, repeat=False, shuffle=True
)
text.build_vocab(train, max_size=max_vocab_size)
train_len = len(text.vocab)
# text.build_vocab(val, max_size=max_vocab_size)
# val_len = len(text.vocab)

#搭建模型
class RNNModel(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size):
        super(RNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, x, hidden):
        emb = self.embed(x)
        output, hidden = self.lstm(emb, hidden)
        # output, (hidden, cell) = self.lstm(emb, hidden)
        # print(hidden)
        # print(hidden[-1])
        # print(hidden[-2])  #这里的lstm是单向的
        out_vocab = self.linear(output.view(-1, output.shape[2]))
            # if num_layers != 1:
            #     hidden = torch.cat((hidden[-1], hidden[-2]), dim=1)
            # else:
            #     hidden = hidden[-1]
        return out_vocab, hidden

    def init_hidden(self, batch_size, requires_grad=True):
        weight = next(self.parameters())
        return (weight.new_zeros((1, batch_size, self.hidden_size), requires_grad=True),
                weight.new_zeros((1, batch_size, self.hidden_size), requires_grad=True))

model = RNNModel(hidden_size=hidden_size, embedding_size=embedding_size, vocab_size=train_len).to(device)

def repackage_hidden(h):
    #两个hidden，分别处理，与之前的切断联系
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


#构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

#评估阶段
def evaluate(model, data):
    model.eval()
    total_loss = 0
    data = iter(data)
    # total_count = len(test_loader)
    total_count1 = 0
    with torch.no_grad():
        hidden = model.init_hidden(batch_size, requires_grad=False)
        for batch in data:
            inputs = batch.text
            targets = batch.target
            inputs = inputs.to(device)
            targets = targets.to(device)
            hidden = repackage_hidden(hidden)

            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            #算每一个batch的总loss，因为交叉熵时平均损失
            total_loss += loss.item() * inputs.size(1)
            total_count1 += inputs.size(1)
            # print('i:', i, 'total_loss:', total_loss, 'total_count:', total_count1)
    #所有测试集上的loss / 所有测试的单词
        loss = total_loss / total_count1
        model.train()
        return loss


#训练阶段
val_losses = []
for epoch in range(num_epochs):
    model.train()
    train_loader = iter(train_loader)
    hidden = model.init_hidden(batch_size)
    for i, batch in enumerate(train_loader):
        inputs = batch.text
        targets = batch.target
        inputs = inputs.to(device)
        targets = targets.to(device)
        hidden = repackage_hidden(hidden)

        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

#第100个batch的损失
        if (i+1) % 100 == 0:
            print('epoch:{}, iteration:{}, train_loss：{}'.format(epoch+1, i+1, loss.item()))

        if (i+1) % 1000 == 0 and i != 0:
            val_loss = evaluate(model, val_loader)
            print('epoch:{}, iteration:{}, test_loss：{}'.format(epoch + 1, i+1, val_loss))
            if len(val_losses) == 0 or val_loss < min(val_losses):
                torch.save(model.state_dict(), 'LM.ckpt')
            else:
                scheduler.step()
            val_losses.append(val_loss)
        # val_loss = evaluate(model, val_loader)
        # print('epoch:{}, iteration:{}, loss'.format(epoch + 1, i+1, val_loss))
        # if len(val_loss) == 0 or val_loss < min(val_losses):
        #     torch.save(model.state_dict(), 'LM.ckpt')
        # else:
        #     scheduler.step()
        # val_losses.append(val_loss)


best_model = RNNModel(hidden_size=hidden_size, embedding_size=embedding_size, vocab_size=len(text.vocab))

best_model.load_state_dict(torch.load('LM.ckpt'))
test_loss = evaluate(best_model, test_loader)
print('perplexity:', np.exp(test_loss))

hidden = best_model.init_hidden(1)
inputs = torch.randint(len(text.vocab), (1, 1), dtype=torch.long)
words = []
for i in range(100):
    outputs, hidden = best_model(inputs, hidden)
    word_weight = outputs.squeeze().exp()
    word_idx = torch.multinomial(word_weight, 1)[0]
    inputs.fill_(word_idx)
    word = text.vocab.itos[word_idx]
    words.append(word)
print(' '.join(words))



