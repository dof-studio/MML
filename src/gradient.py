# gradiant.py
#
# A gradient computation implementation
# From MML Library by Nathmath

import numpy as np

def gradient(func, params, eps=1e-8):
    """
    Computes the numerical gradient of a function with respect to its parameters.
    
    Args:
        func (callable): The function for which to compute the gradient.
        params (numpy.ndarray): The parameters at which to compute the gradient.
        eps (float): A small epsilon for finite differences.
    
    Returns:
        numpy.ndarray: The computed gradient.
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + eps
        f_plus = func(params)
        params[i] = orig - eps
        f_minus = func(params)
        grad[i] = (f_plus - f_minus) / (2 * eps)
        # Restore original value
        params[i] = orig  
    return grad

def numerical_gradient(func, params, eps=1e-8):
    """
    Computes the numerical gradient of a function with respect to its parameters.
    
    Args:
        func (callable): The function for which to compute the gradient.
        params (numpy.ndarray): The parameters at which to compute the gradient.
        eps (float): A small epsilon for finite differences.
    
    Returns:
        numpy.ndarray: The computed gradient.
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + eps
        f_plus = func(params)
        params[i] = orig - eps
        f_minus = func(params)
        grad[i] = (f_plus - f_minus) / (2 * eps)
        # Restore original value
        params[i] = orig  
    return grad
