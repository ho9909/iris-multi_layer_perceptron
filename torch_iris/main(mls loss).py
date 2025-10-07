import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.nn.functional as F
from torch import optim

df = pd.read_csv('iris.csv')


#종이름 번호로 변환: Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2

labels = {'Setosa':[1, 0, 0], 'Versicolor':[0, 1, 0], 'Virginica':[0, 0, 1]}
df['IrisType_num'] = df['variety']
df.IrisType_num = [labels[item] for item in df.IrisType_num]

input_data = df.iloc[:, 0:-2]
input_data = torch.FloatTensor(input_data.to_numpy())
output_data = torch.FloatTensor(df['IrisType_num'].tolist())

data = TensorDataset(input_data, output_data)

#학습 데이터와 테스트 데이터 분할
train_batch_size = 10
number_rows = len(input_data)  # The size of our dataset or the number of rows in excel table.
test_split = int(number_rows * 0.2)
train_split = number_rows - test_split
train_set, test_set = random_split(data, [train_split, test_split])

# Create Dataloader to read the data within batch sizes and put into memory.
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1)

# Define model parameters
input_size = list(input_data.shape)[
    1]  # = 4
learning_rate = 0.01
output_size = len(labels)  # The output is prediction results for three types of Irises.


# Define neural network
class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_size, 30)
        self.layer2 = nn.Linear(30, 10)
        self.layer2 = nn.Linear(10, 8)
        self.layer3 = nn.Linear(8, 6)
        self.layer4 = nn.Linear(6, output_size)

    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = F.relu(self.layer3(x2))
        x4 = self.layer4(x3)
        return F.softmax(x4, dim=1)
    # Instantiate the model


model = Network(input_size, output_size)

# 사용할 디바이스 선택
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("모델 학습 시작", device, "device\n")
model.to(device)    # cpu 또는 gpu 선택

#손실 함수 정의
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습 함수
def train(num_epochs):
    print("Begin training...")

    for epoch in range(1, num_epochs + 1):
        running_train_loss = 0.0

        # 학습 반복
        for data in train_loader:
            # for data in enumerate(train_loader, 0):
            inputs, outputs = data  # get the input and real species as outputs; data is a list of [inputs, outputs]
            predicted_outputs = model(inputs)  # predict output from the model

            train_loss = loss_fn(predicted_outputs.to(torch.float32), outputs.to(torch.float32))  # calculate loss for the predicted output

            optimizer.zero_grad()  # zero the parameter gradients
            train_loss.backward()  # backpropagate the loss
            optimizer.step()  # adjust parameters based on the calculated gradients
            running_train_loss += train_loss.item()  # track the loss value

        # 손실값 계산(loss)
        train_loss_value = running_train_loss / len(train_loader)

            # Print the statistics of the epoch
        print('시행 횟수', epoch, '학습당 로스: %.4f' % train_loss_value)


# Function to test the model
def test():
    # Load the model that we saved at the end of the training loop
    #model = Network(input_size, output_size)

    running_accuracy = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            outputs = outputs.to(torch.float32)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()

        print('Accuracy of the model based on the test set of', test_split,
              'inputs is: %d %%' % (100 * running_accuracy / total))

    # Optional: Function to test which species were easier to predict


def test_species():
    # Load the model that we saved at the end of the training loop

    with torch.no_grad():
        print(len(test_loader))
        for data in test_loader:
            inputs, outputs = data
            predicted_outputs = model(inputs)
            result = predicted_outputs.numpy()
            print(result)
            label = outputs[0]
            print(outputs)


if __name__ == "__main__":
    num_epochs = 100
    train(num_epochs)
    print('Finished Training\n')
    test()
    test_species()
