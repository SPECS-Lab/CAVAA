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

        self.q = 0.8  # ACh level (0 corresponds to shared inhibition and 1 to mutual inhibition) Competition level

        # Initial state
        self.U = []

        # Initial 'dummy' variance
        self.var = 50  # noise

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





class Attractor(object):

    def __init__(self):

        '''
        PARAMETERS
        '''

        self.dt = 0.001

        # couplings
        self.we1 = 5 #self-exitation
        self.we2 = 5
        self.wsi = 4
        self.wmi = 4

        self.q = 0.5  # ACh level (0 corresponds to shared inhibition and 1 to mutual inhibition) Competition level

        # external inputs
        self.I1 = -100
        self.I2 = -100

        # Initial state
        self.U1, self.U2 = [0, 0]

        # Initial 'dummy' variance
        self.var = 50   #noise

        #time decay constants
        self.tau1 = 0.02
        self.tau2 = 0.02

        # activation function
        self.a = 1/22
        self.thr = 15
        self.fmax = 40       


    def sigmoid(self, x):
        return self.fmax / (1 + np.exp(-self.a * (x - self.thr)))

    # this function returns the right hand side of the modified Wilson-Cowan equation
    def WilsonCowan(self, I1, I2):
        

        y1 = ( -self.U1 + self.sigmoid(self.we1 * self.U1 - self.q * self.wmi * self.U2 - (1-self.q) * self.wsi * self.sigmoid(self.U1+self.U2) + I1) + np.random.normal(0,1)*self.var )/self.tau1
        y2 = ( -self.U2 + self.sigmoid(self.we2 * self.U2 - self.q * self.wmi * self.U1 - (1-self.q) * self.wsi * self.sigmoid(self.U1+self.U2) + I2) + np.random.normal(0,1)*self.var )/self.tau2

        
        return np.array([y1, y2])

    def advance(self, I1, I2):


        #self.we1 = 5 * (I1 + 1)
        #self.we2 = 5 * (I2 + 1)
            
        y1, y2 = self.WilsonCowan(I1, I2)
        self.U1 += y1*self.dt
        self.U2 += y2*self.dt
        
        # ReLU correction
        self.U1 *= self.U1>0
        self.U2 *= self.U2>0

        return self.U1, self.U2



