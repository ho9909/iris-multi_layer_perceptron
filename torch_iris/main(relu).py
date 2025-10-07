import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('iris.csv')

labels = {'Setosa':[1, 0, 0], 'Versicolor':[0, 1, 0], 'Virginica':[0, 0, 1]}
df['IrisType_num'] = df['variety']
df.IrisType_num = [labels[item] for item in df.IrisType_num]

input_data = df.iloc[:, 0:-2]
input_data = torch.FloatTensor(input_data.to_numpy())
output_data = torch.FloatTensor(df['IrisType_num'].tolist())

dataset = TensorDataset(input_data, output_data)

train_batch_size = 10
number_rows = len(input_data)
test_split = int(number_rows * 0.2)
train_split = number_rows - test_split
train_set, test_set = random_split(dataset, [train_split, test_split])

train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1)

input_size = list(input_data.shape)[1]
learning_rate = 0.01


class model(nn.Module):
    def __init__(self, input_size):
        super(model, self).__init__()
        self.layer1 = nn.Linear(input_size, 30)
        self.layer2 = nn.Linear(30, 20)
        self.layer3 = nn.Linear(20, 10)
        self.layer4 = nn.Linear(10, 3)

    def forward(self, x):
        x1 = F.relu(self.layer1(x))   # F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))   # F.relu(self.layer2(x1))
        x3 = F.relu(self.layer3(x2))   # F.relu(self.layer3(x2))
        x4 = self.layer4(x3)
        return F.softmax(x4, dim=1)


model = model(input_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

loss = nn.CrossEntropyLoss()     #nn.CrossEntropyLoss()     #
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # optim.Adam(model.parameters(), lr=learning_rate)


def train(num_epochs):
    print("학습 시작")

    for epoch in range(1, num_epochs+1):
        total_train_loss = 0.0
        for data in train_loader:
            inputs, outputs = data
            result = model(inputs)

            train_loss = loss(result, outputs)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()

        train_loss_value = total_train_loss / len(train_loader)
        if epoch % 10 == 0:
            print('시행 횟수', epoch, '학습당 로스: %.4f' % train_loss_value)


def test():
    total_loss = 0

    for data in test_loader:
        inputs, outputs = data
        outputs_data = outputs
        result = model(inputs)
        print("결과값 : ", outputs_data[0].tolist(), "실제 값 : ", np.round(result[0].tolist(), 3))
        train_loss = loss(result, outputs)
        total_loss += train_loss.item()

    total_loss = total_loss / len(test_loader)
    print("전체 로스값 : ", total_loss)


if __name__ == "__main__":
    num_epochs = 1000
    train(num_epochs)
    # t1 = [0.2239, 0.1405, 0.0795, 0.0329, 0.0230, 0.0196, 0.0172, 0.0177]
    # t2 = [0.2177, 0.0229, 0.0169, 0.0173, 0.0201, 0.0113, 0.0148, 0.0147]
    # t3 = [1.0939, 0.6757, 0.6062, 0.5946, 0.5878, 0.5810, 0.5854, 0.5821]
    # t4 = [1.0812, 0.6080, 0.6005, 0.6228, 0.5797, 0.6094, 0.6042, 0.5767]
    # x = [1, 100, 200, 300, 400, 500, 600, 700]
    # plt.plot(x, t1, label='Mseloss-SGD')
    # plt.plot(x, t2, 'r-', label='Mseloss-Adam')
    # plt.plot(x, t3, '-g', label='CrossEnroyLoss-SGD')
    # plt.plot(x, t4, 'y-', label='CrossEnroyLoss-Adam')
    #
    # plt.legend()
    plt.show()
    print("테스트 시작")
    test()

