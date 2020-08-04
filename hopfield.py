import numpy as np

class Hopfield(object):
    def __init__(self, theta, N):
        self.theta = theta
        self.dim = N
        self.W = np.zeros([self.dim, self.dim])

    def update(self, train_data):
        #train_data = epoch * dim
        for pattern in train_data:
            for i in range(len(pattern)):
                for j in range(0, len(pattern)):
                    self.W[i][j] += pattern[i] * pattern[j]
        self.W /= len(train_data)
        for i in range(self.dim):
            for j in range(self.dim):
                if i > j:
                    self.W[i][j] = self.W[j][i]
                if i == j:
                    self.W[i][j] = 0
    
    def sgn(self, u):
        if u - self.theta >= 0:
            return 1
        else:
            return -1

    def energy(self, train_list):
        V = 0
        for i in range(len(train_list)):
            for j in range(len(train_list)):
                V += (-1/2) * self.W[i][j] * train_list[i] * train_list[j] + self.theta * train_list[i]
        return V

    def __call__(self, x_0):
        x_1 = np.zeros(len(x_0))
        for i in range(len(x_0)):
            sum = 0
            for j in range(len(x_0)):
                sum += self.W[i][j] * x_0[j]
            x_1[i] = self.sgn(sum)
        return x_1