import numpy as np

class Multiattractor(object):

    def __init__(self):

        '''
        PARAMETERS
        '''

        self.dt = 0.001

        # couplings
        self.we = 5  # self-exitation
        self.wsi = 4
        self.wmi = 4

        self.q = 0.8  # ACh level (0 corresponds to shared inhibition and 1 to mutual inhibition) Competition level - 0.2

        # Initial state
        self.U = []

        # Initial 'dummy' variance
        self.var = 50  # noise - 50

        # time decay constants
        self.tau = 0.02

        # activation function
        self.a = 1 / 22
        self.thr = 15
        self.fmax = 40


    def sigmoid(self, x):
        return self.fmax / (1 + np.exp(-self.a * (x - self.thr)))

    # this function returns the right hand side of the modified Wilson-Cowan equation
    def WilsonCowan(self, input_list):

        y_list = []

        for i in range(len(input_list)):
            I = input_list[i]
            wU = self.we * self.U[i]
            inh1 = self.q * self.wmi * (sum(self.U) - self.U[i])
            inh2 = (1 - self.q) * self.wsi * self.sigmoid(sum(self.U))
            noise = np.random.normal(0, 1 * self.var)

            y_list.append((-self.U[i] + self.sigmoid(wU - inh1 - inh2 + I) + noise) / self.tau)

        return np.array(y_list)

    def advance(self, *input_list):

        if self.U == []:
            self.U = [0] * len(input_list)

        y = self.WilsonCowan(input_list)

        for i in range(len(input_list)):
            self.U[i] += y[i] * self.dt
            self.U[i] *= self.U[i] > 0

        return self.U