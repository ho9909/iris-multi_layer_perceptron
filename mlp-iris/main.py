import random
import copy
import numpy as np
import pandas as pd

df = pd.read_csv('iris.csv')


class multi_perceptron:
    def __init__(self):

        self.w_node = 10
        self.w1_node = 8
        self.w2_node = 6
        self.out_node = 3
        self.input_node = 4

        self.h_loss = np.array([float(0) for col in range(self.w_node)])
        self.h1_loss = np.array([float(0) for col in range(self.w1_node)])
        self.h2_loss = np.array([float(0) for col in range(self.w2_node)])
        self.out_loss = np.array([float(0) for col in range(self.out_node)])

        self.out_put = np.array([float(0) for col in range(self.out_node)])
        
        self.h2_layer = np.array([float(0) for col in range(self.w2_node)])
        self.h1_layer = np.array([float(0) for col in range(self.w1_node)])
        self.h_layer = np.array([float(0) for col in range(self.w_node)])

        self.w3 = self.make_weight(self.w2_node, self.out_node)
        self.w2 = self.make_weight(self.w1_node, self.w2_node)
        
        self.w1 = self.make_weight(self.w_node, self.w1_node)
        self.w = self.make_weight(self.input_node, self.w_node)

        self.input_data = []
        self.test_data = []

        self.bias = 1
        self.learning_rate = 0.2
        self.epoch = 10000
        self.tss = 0

        self.error_data = np.array([float(0) for col in range(self.out_node)])

        a_list = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
        result = [[], [], [], []]
        test = [[],[],[],[]]

        for i in range(len(a_list)):
            a = np.array(df[a_list[i]])
            result[i], test[i] = self.normalization(a, a.max(), a.min())

        self.input_data = self.make_list(result)
        self.test_data = self.make_list(test)

        for i in range(len(self.input_data)):
            self.input_data[i][0] = result[0][i]
            self.input_data[i][1] = result[1][i]
            self.input_data[i][2] = result[2][i]
            self.input_data[i][3] = result[2][i]

        for i in range(len(self.test_data)):
            self.test_data[i][0] = test[0][i]
            self.test_data[i][1] = test[1][i]
            self.test_data[i][2] = test[2][i]
            self.test_data[i][3] = test[3][i]

        self.run()

    def data(self, a):
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

    def step_function(self, result):
        for i in range(len(result)):
            if result[i] > 0.5:
                result[i] = 1

            else:
                result[i] = 0
        return result

    def normalization(self, a, b, c):
        for i in range(len(a)):
            a[i] = round((a[i] - c) / (b - c), 3)

        input_data, test_data = self.data(a)

        return input_data, test_data

    def make_list(self, data):
        result = []
        for i in range(len(data[0])):
            result.append([])
            for j in range(4):
                result[i].append(0)

        return result

    def make_weight(self, pre_node, next_node):
        w = np.array([[random.random()-0.5 for col in range(pre_node+1)]for row in range(next_node)])

        return w

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def forward_propagation(self, pre_layer, next_layer, weight):
        for i in range(len(next_layer)):
            next_layer[i] = 0

            for j in range(len(pre_layer)):
                next_layer[i] += float(pre_layer[j]) * weight[i][j]
            next_layer[i] += self.bias * weight[i][len(pre_layer)]
            next_layer[i] = self.sigmoid(next_layer[i])

    def error(self, error, target, output):
        for i in range(self.out_node):
            error[i] = (target[i] - output[i])

    def first_delta(self, first_delta, output, error):
        for i in range(len(error)):
            first_delta[i] = error[i] * output[i] * (1-output[i])

    def delta(self, weight, layer, delta, next_node, pre_node):
        for i in range(next_node):
            for j in range(pre_node):
                weight[i][j] += self.learning_rate * layer[j] * delta[i]
            weight[i][pre_node] += self.learning_rate * self.bias * delta[i]

    def back_propagation(self, pre_delta, next_delta, layer, weight):
        for i in range(len(next_delta)):
            next_delta[i] = 0
            for j in range(len(pre_delta)):
                next_delta[i] += pre_delta[j] * weight[j][i]
            next_delta[i] = next_delta[i] * layer[i] * (1 - layer[i])

    def pow_pattern_error(self, error):
        pattern_error = 0
        for i in range(self.out_node):
            error[i] = pow(error[i], 2)
            pattern_error += error[i]
        return pattern_error

    def run(self):
        target = [[1,0,0],[0,1,0], [0,0,1]]
        for epoch in range(self.epoch):
            self.tss = 0
            count = 0
            cnt = 0
            print("epoch :", epoch)
            for j in range(len(self.input_data)):
                self.forward_propagation(self.input_data[j], self.h_layer, self.w)
                self.forward_propagation(self.h_layer, self.h1_layer, self.w1)
                self.forward_propagation(self.h1_layer, self.h2_layer, self.w2)
                self.forward_propagation(self.h2_layer, self.out_put, self.w3)

                self.error(self.error_data, target[count], self.out_put)

                self.first_delta(self.out_loss, self.out_put, self.error_data)
                self.back_propagation(self.out_loss, self.h2_loss, self.h2_layer, self.w3)
                self.back_propagation(self.h2_loss, self.h1_loss, self.h1_layer, self.w2)
                self.back_propagation(self.h1_loss, self.h_loss, self.h_layer, self.w1)

                self.delta(self.w3, self.h2_layer, self.out_loss, self.out_node, self.w2_node)
                self.delta(self.w2, self.h1_layer, self.h2_loss, self.w2_node, self.w1_node)
                self.delta(self.w1, self.h_layer, self.h1_loss, self.w1_node, self.w_node)
                self.delta(self.w, self.input_data[j], self.h_loss, self.w_node, self.input_node)

                temp = self.pow_pattern_error(self.error_data) / self.out_node
                self.tss += temp
                #print(cnt, "번째 타겟값 : ", target[count], "결과값 :", self.step_function(self.out_put))
                cnt += 1
                if cnt % 40 == 0:
                    count += 1
                if cnt == 120:
                    count = 0


            print("tss", round(self.tss, 4),"\n")
            if self.tss < 0.2:
                break
        self.test()

    def test(self):
        cnt = 1
        for i in range(len(self.test_data)):
            self.forward_propagation(self.test_data[i], self.h_layer, self.w)
            self.forward_propagation(self.h_layer, self.h1_layer, self.w1)
            self.forward_propagation(self.h1_layer, self.h2_layer, self.w2)
            self.forward_propagation(self.h2_layer, self.out_put, self.w3)
            print(cnt, "번째 결과", self.step_function(self.out_put))
            cnt += 1
            if cnt == 11 or cnt == 21:
                print("\n")

a = multi_perceptron()