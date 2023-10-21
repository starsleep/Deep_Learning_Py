import numpy as np

class AcrivationFunc:

    #def __init__(self,):
            #self. = 0

    def Sigmoid_Func(Mat):
        return (1 / (1 + np.exp(-Mat)))
    
    def Step_Func(Mat):
        return np.array(Mat > 0,dtype= np.int)
    
    def ReLU_Func(Mat):
        return np.maximum(0,Mat)