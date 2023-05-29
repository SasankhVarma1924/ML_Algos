import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def si(z):
    g = (1/(1+np.exp(-z)))
    return g

def z_normalization(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    xc = (x-mu)/sigma
    return xc,mu,sigma

class LogisticReg:
    x_train = None
    y_train = None
    b = 0.0
    lr = 0.0
    def __init__(self, x_data, y_data, l):
        self.x_train = x_data
        self.y_train = y_data
        self.lr = l
        self.m_values = np.zeros((x_data.shape[1]))

    def print_coeff(self):
        print(self.m_values, self.b)

    def fit(self, iter):
        for i in range(iter):
            self.gradient_descent()

    def predict(self, x):
        fx = np.dot(x, self.m_values) + self.b
        if si(fx) > 0.5:
            return 1
        else:
            return 0


    def predicted_values(self):
        m, n = self.x_train.shape
        pred = np.zeros((m,))
        for i in range(m):
            fx = np.dot(self.x_train[i], self.m_values) + self.b
            if si(fx) > 0.5:
                pred[i] = 1
            else:
                pred[i] = 0
        return pred


    def cost_function(self):
        m, n = self.x_train.shape
        cost = 0.0
        for i in range(m):
            fx = np.dot(self.x_train[i], self.m_values) + self.b
            fx = si(fx)
            cost += (-self.y_train[i] * np.log(fx)) - (1 - self.y_train[i]) * np.log(1 - fx)

        return cost / m

    def gradient_descent(self):
        m, n = self.x_train.shape
        dj_dm = np.zeros(self.x_train.shape[1])
        dj_db = 0.0
        for i in range(m):
            fx = np.dot(self.x_train[i], self.m_values) + self.b
            err = si(fx) - self.y_train[i]
            for j in range(n):
                dj_dm[j] += err * self.x_train[i][j]
            dj_db += err

        self.m_values -= (self.lr * (dj_dm / m))
        self.b -= (self.lr * (dj_db/m))




df = pd.read_csv("../DataSets/diabetes2.csv")
df_np = df.to_numpy()
x_train = df_np[:,:-1]
y_train = df_np[:,-1:]
print(x_train)

x_train, x_mu, x_sigma = z_normalization(x_train)

o = LogisticReg(x_train, y_train, 0.1)
o.fit(1000)
o.print_coeff()

person = np.array([1,103,30,38,83,43.3,0.183,33])
z_person = (person - x_mu)/x_sigma
print(o.predict(z_person))

