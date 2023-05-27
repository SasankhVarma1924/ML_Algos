import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import array as arr

def z_normalization(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    xc = (x-mu)/sigma
    return xc,mu,sigma

class MultipleLinearReg:
    x_train = None
    y_train = None
    m_values = None
    b = 0.0
    lr = 0.0
    cost_iter = arr.array('f')

    def __init__(self, x_data, y_data, l_rate):
        self.x_train = x_data
        self.y_train = y_data
        self.lr = l_rate
        self.m_values = np.zeros((self.x_train.shape[1]))


    def print_coeff(self):
        print(self.m_values, self.b)


    def fit(self, iter):
        for i in range(iter):
            self.gradient_descent()
            if i % 50 == 0:
                self.cost_iter.append(self.cost_funtion())


    def plot_cost_iter(self):
        n = len(self.cost_iter)
        plt.plot([x*50 for x in range(n)], self.cost_iter)
        plt.show()

    def predict(self, x_data):
        return np.dot(x_data, self.m_values) + self.b

    def predictied_values(self):
        m,n = self.x_train.shape
        y_pred = np.zeros((m,))
        for i in range(m):
            y_pred[i] = np.dot(self.x_train[i],self.m_values) + self.b
        return y_pred
    def cost_funtion(self): #mean squared error
        m, n = self.x_train.shape
        cost = 0.0
        for i in range(m):
            cost += (1/m) * (self.y_train[i] - np.dot(self.x_train[i], self.m_values) + self.b)
        return cost
    def gradient_descent(self):
        m, n = self.x_train.shape
        dj_dm = np.zeros((n,))
        dj_db = 0.0

        for i in range(m):
            err = (np.dot(self.x_train[i], self.m_values) + self.b) - self.y_train[i]
            for j in range(n):
                dj_dm[j] += err * x_train[i][j]
            dj_db += err

        dj_dm = dj_dm / m
        dj_db = dj_db / m
        self.m_values = self.m_values - (dj_dm * self.lr)
        self.b -= (dj_db * self.lr)

df = pd.read_csv("../DataSets/houses.csv")

df_np = df.to_numpy()
x_train = df_np[:,:-1]
y_train = df_np[:,-1:]

x_train,x_mu,x_sigma = z_normalization(x_train)

o = MultipleLinearReg(x_train, y_train,  0.1)
o.fit(1000)
o.print_coeff()

x_house = np.array([1200, 3, 1, 40])
z_house = (x_house - x_mu)/x_sigma
print(o.predict(z_house))
