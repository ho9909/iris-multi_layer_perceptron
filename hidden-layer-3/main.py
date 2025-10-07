# Multi-Perceptron
import random
import copy
import numpy as np
import pandas as pd

df = pd.read_csv('iris.csv')


def data(a):
    b = np.arange(0, 3, 0.1)
    b[0:10] = a[40:50]
    b[10:20] = a[90:100]
    b[20:30] = a[140:150]
    cnt = 40
    for i in range(10):
        a = np.delete(a, cnt)
    cnt = 80
    for i in range(10):
        a = np.delete(a, cnt)
    cnt = 120
    for i in range(10):
        a = np.delete(a, cnt)

    return a, b


class multi_perceptron:
    def __init__(self):
        self.a = 0
        self.cnt = 0
        self.cur = 1
        self.target = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 목표값
        self.loss = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self.result = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        # input Weight
        self.w1 = self.make_weight(4, 10)
        print(self.w1)
        self.w2 = self.make_weight(10, 8)

        self.w3 = self.make_weight(8, 6)

        self.w4 = self.make_weight(6, 3)

        self.bw = self.make_weight(10)

        self.b = 1
        self.running = 0.5  # 이동 값
        self.epoch = 100  # 반복 횟수
        self.count = 0
        self.tss = 0

        a = np.array(df['sepallength'])
        b = np.array(df['sepalwidth'])
        c = np.array(df['petallength'])
        d = np.array(df['petalwidth'])

        nor_a = self.normalization(a, a.max(), a.min())
        nor_b = self.normalization(b, b.max(), b.min())
        nor_c = self.normalization(c, c.max(), c.min())
        nor_d = self.normalization(d, d.max(), d.min())

        self.input, self.d = data(nor_a)
        self.input1, self.d1 = data(nor_b)
        self.input2, self.d2 = data(nor_c)
        self.input3, self.d3 = data(nor_d)

        self.r_input = self.make_weight(120, 4)

        self.r_test = self.make_weight(30, 4)

        for i in range(len(self.r_input)):
            self.r_input[i][0] = self.input[i]
            self.r_input[i][1] = self.input1[i]
            self.r_input[i][2] = self.input2[i]
            self.r_input[i][3] = self.input3[i]

        for j in range(len(self.r_test)):
            self.r_test[j][0] = self.d[j]
            self.r_test[j][1] = self.d1[j]
            self.r_test[j][2] = self.d2[j]
            self.r_test[j][3] = self.d3[j]

        self.Run()      # 시작.

    def make_weight(self, len_a, len_b=1):
        w = []
        # bias weight 삽입시
        if len_b == 1:
            for i in range(len_a):
                w.append(random.random())

        # input weight ~ ouput weight 까지
        else:
            for i in range(len_a):
                w.append([])
                for j in range(len_b):
                    w[i].append(random.random())
        return w

    def threshold(self, result):
        for i in range(len(result)):
            if result[i] < 0.5:
                result[i] = 0
            else:
                result[i] = 1
        return result

    def sigmoid(self, result):
        z = np.exp(-result)
        sig = 1 / (1+z)
        return sig


    def relu(self, result):
        if result <= 0:
            return 0
        else:
            return result

    def weight_sum(self, x, layer, number, weight):  # layer = 층, number = 노드 번호
        number = number - 1
        result = 0
        temp_x = copy.deepcopy(x)
        w = copy.deepcopy(weight)
        # 웨이트 넣기
        if layer == 1:
            temp_x.append(self.b)
            w.append(self.bw)
            for i in range(len(w)):
                result += temp_x[i] * w[i][number]
        elif layer == 2:
            for i in range(len(w)):
                result += temp_x[i] * w[i][number]
        elif layer == 3:
            for i in range(len(w)):
                result += temp_x[i] * w[i][number]
        elif layer == 4:
            for i in range(len(w)):
                result += temp_x[i] * w[i][number]
        else:
            print("error : layer 입력 에러")
            return
        return result

    def learning(self, x):
        # 1층 연산

        for i in range(len(self.result)):
            self.result[i] = self.initialization(self.result[i])

        # hidden 1층
        for j in range(len(self.result[0])):
            self.result[0][j] = self.sigmoid(self.weight_sum(x, 1, j+1, self.w1))
        # hidden 2층
        for k in range(len(self.result[1])):
            self.result[1][k] = self.sigmoid(self.weight_sum(self.result[0], 2, k+1, self.w2))
        # hidden 3층
        for a in range(len(self.result[2])):
            self.result[2][a] = self.sigmoid(self.weight_sum(self.result[1], 3, a+1, self.w3))
        # output
        for r in range(len(self.result[3])):
            self.result[3][r] = self.sigmoid(self.weight_sum(self.result[2], 4, r+1, self.w4))

    def delta_out(self, target, result):
        delta = (target - result) * (1-result) * result
        return delta

    def initialization(self, loss):
        for i in range(len(loss)):
            loss[i] = 0.0

        return loss

    def backpropagation(self, x):
        out_delta = [0.0, 0.0, 0.0]
        for i in range(len(self.loss)):
            self.loss[i] = self.initialization(self.loss[i])

        # output loss
        for i in range(len(self.result[3])):
            self.loss[3][i] = (self.target[self.count][i] - self.result[3][i])*(1-self.result[3][i])*self.result[3][i]

        for i in range(len(out_delta)):
            out_delta[i] = self.delta_out(self.target[self.count][i], self.result[3][i])

        # hidden layer loss
        for i in range(len(self.loss[2])):
            for k in range(len(self.w4[i])):
                self.loss[2][i] += (out_delta[k] * self.w4[i][k])
            self.loss[2][i] = self.loss[2][i] * (1 - self.result[2][i]) * self.result[2][i]

        for i in range(len(self.loss[1])):
            for k in range(len(self.w3[i])):
                self.loss[1][i] += (self.loss[2][k] * self.w3[i][k])
            self.loss[1][i] = self.loss[1][i] * (1 - self.result[1][i]) * self.result[1][i]

        for i in range(len(self.loss[0])):
            for k in range(len(self.w2[i])):
                self.loss[0][i] += (self.loss[1][k] * self.w2[i][k])
            self.loss[0][i] = self.loss[0][i] * (1 - self.result[0][i]) * self.result[0][i]

        # weight 업데이트
        for i in range(len(self.w4)):
            self.weight_update(self.result[2][i], self.loss[3], 4, i+1)

        for i in range(len(self.w3)):
            self.weight_update(self.result[1][i], self.loss[2], 3, i+1)

        for i in range(len(self.w2)):
            self.weight_update(self.result[0][i], self.loss[1], 2, i+1)

        for i in range(len(self.w1)):
            self.weight_update(x[i], self.loss[0], 1, i+1)
            self.weight_update(self.b, self.loss[0], 1, i + 1, 1)

        for i in range(3):
            z = pow(self.loss[3][i], 2)
            self.tss += z


    def weight_update(self, x, loss, layer, number, cnt=0):
        number -= 1
        # Weight Update
        if layer == 1 and cnt == 0:
            for i in range(len(loss)):  # 받는 Weight 갯수
                self.w1[number][i] = (self.w1[number][i]) + self.running * loss[i] * x
        elif layer == 1 and cnt == 1:
            self.bw[number] = (self.bw[number]) + self.running * loss[number] * x

        elif layer == 2:
            for i in range(len(loss)):
                self.w2[number][i] = (self.w2[number][i]) + self.running * loss[i] * x

        elif layer == 3:
            for i in range(len(loss)):
                self.w3[number][i] = (self.w3[number][i]) + self.running * loss[i] * x

        elif layer == 4:
            for i in range(len(loss)):
                self.w4[number][i] = (self.w4[number][i]) + self.running * loss[i] * x

    def Run(self):
        for epoch in range(self.epoch):
            self.cnt = 0
            self.tss = 0
            print("epoch :", epoch + 1)
            for i in range(len(self.r_input)):
                self.learning(self.r_input[i])     # 합 구하기
                self.backpropagation(self.r_input[i])  # 역전파
                #print("(", round(self.r_input[i][0], 3), ",", round(self.r_input[i][1], 3), ",", round(self.r_input[i][2], 3), ",", round(self.r_input[i][3], 3), ")", end=" ")
                #print("target :", self.target[self.count], "output :", self.learning([self.r_input[i][0], self.r_input[i][1], self.r_input[i][2], self.r_input[i][3]]), end=" ")
                #print("th(output) :", self.threshold(self.learning([self.r_input[i][0], self.r_input[i][1], self.r_input[i][2], self.r_input[i][3]])))
                self.cnt += 1
                if self.cnt % 40 == 0:
                    self.count += 1
                if self.cnt == 120:
                    self.count = 0
            print("tss", round(self.tss/120, 4))

        print("\n")
        print("=========================================")
        print("test data")
        self.test()

    def test(self):
        cur = 0
        cnt = 0
        for i in range(len(self.r_test)):
            print("(", round(self.r_test[i][0], 3), ",", round(self.r_test[i][1], 3), ",",
                  round(self.r_test[i][2], 3), ",", round(self.r_test[i][3], 3), ")", end=" ")
            self.learning(self.r_test[i])
            print("target :", self.target[cnt], end=" ")
            print("output : ", self.result[3], end=" ")
            print("result :", self.threshold(self.result[3]))
            cur += 1
            if cur % 10 == 0:
                cnt += 1
                print("\n")

    def normalization(self, a, b, c):
        for i in range(len(a)):
            a[i] = round((a[i] - c) / (b - c), 3)

        return a


multi_perceptron()
