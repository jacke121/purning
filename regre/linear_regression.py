import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class LinearRegression():

    def __init__(self):

        self.weights = None
        self.bias = None

    def fit(self, X, y):

        num_epochs = 10000
        N,K = X.shape
        inputs = Variable(X)
        actual = Variable(y)

        criterion = nn.MSELoss()
        linear = torch.nn.Linear(K, 1, bias=True)
        optimizer = optim.Adam(linear.parameters())

        for epoch in range(num_epochs):

            outputs = linear(inputs)
            loss = criterion(outputs, actual)
            print('loss',float(loss.cpu().data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(linear.weight)

        self.weight = linear.weight.data.numpy()[0]
        self.bias = linear.bias.data.numpy()[0]


    def predict(self, X):

        return self.bias + X.dot(self.weight)



if __name__ == "__main__":

    N = 5000
    K = 7
    filename='lograndtmp.txt'
    X=[]
    y=[]
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            E_tmp = [float(i) for i in lines.split(' ')]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            X.append(E_tmp[:7])
            y.append(E_tmp[-1])
    # mean = np.random.normal(5, .2, size = K)

    # X = np.random.multivariate_normal(mean = datas, cov = np.identity(K), size = (N))
    B = np.random.uniform( size = K)
    # y = X.dot(B).reshape(N,1)# + np.random.normal(0,1, size = N)

    X_torch = torch.from_numpy(np.asarray(X)).float()
    y_torch = torch.from_numpy(np.asarray(y)).float()

    Linear = LinearRegression()
    Linear.fit(X_torch, y_torch)

    print("actual Beta", B)