import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import array as arr


def z_normalization(x):
    mu = np.mean(x,axis=0)
    sigma = np.var(x,axis=0)
    xc = (x-mu)/sigma
    return xc,mu,sigma


class SingleLinearReg:
    x_train = None
    y_train = None
    m = 0.0
    b = 0.0
    lr = 0.0
    cost_iter = arr.array("f")


    def __init__(self, x_data, y_data, l_rate):
        self.x_train = x_data
        self.y_train = y_data
        self.lr = l_rate


    def fit(self, iter):
        for i in range(iter):
            self.gradient_descent()
            if i % 10 == 0:
                self.cost_iter.append(self.cost_funtion())



    def plot_data(self):
        plt.scatter(self.x_train, self.y_train,c="blue")
        plt.plot(self.x_train, self.predicted_values(),'r')
        plt.show()


    def plot_cost_iter(self):
        n = len(self.cost_iter)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.plot([x*10 for x in range(n)], self.cost_iter,'r')
        plt.show()


    def predicted_values(self):
        y_pred = np.array([(self.m * x + self.b) for x in self.x_train])
        return y_pred


    def predict(self,x_value):
        return self.m * x_value + self.b


    def print_coeff(self):
        print(self.m,self.b)


    def cost_funtion(self):
        pass #mean sqaured error
        n = len(self.x_train)
        cost = (1/n) * (self.y_train.sum() - ((self.x_train * self.m) + self.b).sum())
        return cost


    def gradient_descent(self):
        n = len(self.x_train)
        m_grad = 0.0
        b_grad = 0.0
        for i in range(n):
            m_grad += (-2/n) * self.x_train[i] * (self.y_train[i] - (self.m * self.x_train[i] + self.b))
            b_grad += (-2/n) * (self.y_train[i] - (self.m * self.x_train[i] + self.b))
        self.m -= (m_grad * self.lr)
        self.b -= (b_grad * self.lr)


df = pd.read_csv("../DataSets/houses.csv")

df = df.drop(['bedrooms', 'floors', 'age'],axis=1)

x_train = np.array(df["sqft"])
y_train = np.array(df["price"])

x_train, x_mu, x_sigma = z_normalization(x_train)

o = SingleLinearReg(x_train, y_train, 0.1)
o.fit(1000)
o.plot_cost_iter()

x_house = np.array([1200])
z_house = (x_house - x_mu)/x_sigma

print(o.predict(z_house))


