# nn_base.py
#
# A Deep Neural Network Abstract Module Base Class
# From MML Library by Nathmath

from .baseml import MLBase
from .baseml import Regression, Classification


# Deep Neural Network Abstract Module Base Class
class nn_Base(Regression, Classification):
    
    __attr__ = "MML.nn_Base"    
    
    def __init__(self):
        super().__init__()
        
    def zero_grad(self):
        """
        Set all gradients of all parameters to zero.
        It will clear the gradients accumulated and restore when a new batch starts.
        """
        raise NotImplementedError("zero_grad() is not implemented in nn_Base")

    def train(self):
        """
        Set module to training mode.
        It will affect dropouts and enable gradients calculation.
        """
        raise NotImplementedError("train() is not implemented in nn_Base")

    def eval(self):
        """
        Set module to evaluation mode.
        It will disable dropouts and disable gradients calculation.
        """
        raise NotImplementedError("eval() is not implemented in nn_Base")

    def parameters(self):
        """
        Return an iterator of all Parameters in this module (includes children)
        """
        raise NotImplementedError("parameters() is not implemented in nn_Base")
        
    def forward(self):
        """
        Perform a forward propagation to calculate the loss.
        """
        raise NotImplementedError("forward() is not implemented in nn_Base")
        
    def backward(self):
        """
        Perform a backward propagation to compute gradients for updating weights.
        """
        raise NotImplementedError("backward() is not implemented in nn_Base")
        
    def __repr__(self):
        return "nn_Base(Deep Neural Network Abstract Module Base Class)."

