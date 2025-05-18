# nn_activation.py
#
# Neural Network Activation Function Collection
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from typing import Any, Literal
    
from .objtyp import Object
from .tensor import Tensor

from .nn_parameter import nn_Parameter
from .nn_module import nn_Module


# Implementation of ReLU Activation
class nn_Activation_ReLU(nn_Module):
    """
    ReLU activation function.
    
    The Rectified Linear Unit (ReLU) is a widely used activation function 
    defined by the formula: f(x) = \max(0, x). This function outputs the 
    input value if it is positive, and zero otherwise. ReLU is celebrated for its 
    computational efficiency and ability to mitigate vanishing gradient problems 
    during backpropagation, making it a cornerstone in modern deep learning architectures.
    
    Formula: f(x) = max(0, x)
    
    """
    
    __attr__ = "MML.nn_Activation_ReLU"
    
    def __init__(self, 
                 *,
                 module_name: str = "nn_Activation_ReLU", 
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        An ReLU activation function.

        Parameters:
            module_name: str, The name of the module instance.
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """
        
        super().__init__(module_name = module_name, backend = backend, dtype = dtype, device = device, autograd = autograd)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the Rectified Linear Unit (ReLU) activation function to the input tensor.

        This method computes the element-wise ReLU activation, which outputs the input if it is positive,
        and zero otherwise. It is a fundamental non-linearity in neural networks, enabling the model
        to learn complex patterns by introducing non-linearities. The input is saved for backward propagation
        to compute gradients during training.

        Args:
            x (Tensor): Input tensor of any shape. The ReLU operation is applied element-wise.

        Returns:
            Tensor: Output tensor after applying the ReLU activation, with the same shape as the input.

        Raises:
            ValueError: If the input `x` is not a valid MML.Tensor object.
        """
                
        # Type check, x must be an instance of Tensor
        if isinstance(x, Tensor) == False:
            raise ValueError(f"In performing forward(), input `x` must be in a MML `Tensor` format but you have {type(x)}")
        
        # Save input for backward
        self.__setattr__("input", x)

        # Apply ReLU to the input data
        return Tensor.where_as(x.data > 0, x.data, 0, backend=self.backend, dtype=self.dtype, device=self.device)

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a module during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            grad_output (Tensor): Gradient tensor resulting from the output of the layer, used as input for backpropagation.

        Returns:
            Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers.

        Raises:
            ValueError: If `grad_output` is not a valid MML.Tensor object..
        """
        
        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            return None
        
        # Type check, grad_output must be an instance of Tensor
        if isinstance(grad_output, Tensor) == False:
            raise ValueError(f"In performing backward(), `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")
        
        # Pass gradient only where input was positive
        grad_input = Tensor.where_as(self.input.data <= 0, 0, grad_output.data, backend=self.backend, dtype=self.dtype, device=self.device)
        return grad_input
    
    def __repr__(self):
        return "nn_Activation_ReLU(ReLU Activation Function)."
    

# Alias for nn_Activation_ReLU
ReLU = nn_Activation_ReLU


# Implementation of Leaky ReLU Activation
class nn_Activation_LeakyReLU(nn_Module):
    """
    Leaky ReLU activation with a small slope for negative inputs.
    
    The Leaky Rectified Linear Unit (Leaky ReLU) is a variant of the ReLU 
    activation function that allows a small, non-zero gradient when the input 
    is negative. This helps mitigate the "dying ReLU" problem where neurons 
    become inactive and cease to learn. The function is defined as:
    
    Formula: f(x) = max(0, x, α*x), where α is a small positive slope (typically 0.01).
    
    """
    
    __attr__ = "MML.nn_Activation_LeakyReLU"
    
    def __init__(self, 
                 leaky_slope: float = 0.01,
                 *,
                 module_name: str = "nn_Activation_ReLU", 
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        A Leaky ReLU activation function.

        Parameters:
            leaky_slope: float, The slope of negative values.
            module_name: str, The name of the module instance.
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.leaky_slope: float, The slope applied to negative values.
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """
        
        super().__init__(module_name = module_name, backend = backend, dtype = dtype, device = device, autograd = autograd)
    
        # Record the leaky slope as a non-Parameter attribute
        self.__setattr__("leaky_slope", leaky_slope)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the Leaky Rectified Linear Unit (Leaky ReLU) activation function to the input tensor.

        This method computes the element-wise Leaky ReLU activation, which outputs the input if it is positive,
        and a small negative slope multiplied by the input otherwise. This variant of ReLU mitigates the "dying ReLU"
        problem by allowing a controlled negative slope, improving gradient flow for negative inputs. The input is saved
        for backward propagation to compute gradients during training.

        Args:
            x (Tensor): Input tensor of any shape. The Leaky ReLU operation is applied element-wise.

        Returns:
            Tensor: Output tensor after applying the Leaky ReLU activation, with the same shape as the input.

        Raises:
            ValueError: If the input `x` is not a valid MML.Tensor object.
        """
                
        # Type check, x must be an instance of Tensor
        if isinstance(x, Tensor) == False:
            raise ValueError(f"In performing forward(), input `x` must be in a MML `Tensor` format but you have {type(x)}")
        
        # Save input for backward
        self.__setattr__("input", x)

        # Apply Leaky ReLU to the input data
        return Tensor.where_as(x.data > 0, x.data, x.data * self.leaky_slope, backend=self.backend, dtype=self.dtype, device=self.device)

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a module during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            grad_output (Tensor): Gradient tensor resulting from the output of the layer, used as input for backpropagation.

        Returns:
            Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers.

        Raises:
            ValueError: If `grad_output` is not a valid MML.Tensor object..
        """
        
        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            return None
        
        # Type check, grad_output must be an instance of Tensor
        if isinstance(grad_output, Tensor) == False:
            raise ValueError(f"In performing backward(), `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")
        
        # Pass gradient only where input was positive
        grad_input = Tensor.where_as(self.input.data <= 0, grad_output.data * self.leaky_slope, grad_output.data, backend=self.backend, dtype=self.dtype, device=self.device)
        return grad_input
    
    def __repr__(self):
        return f"nn_Activation_LeakyReLU(Leaky ReLU Activation Function with alpha = {self.leaky_slope})."
    
      
# Alias for nn_Activation_LeakyReLU
LeakyReLU = nn_Activation_LeakyReLU 


# Implementation of Sigmoid Activation
class nn_Activation_Sigmoid(nn_Module):
    """
    Sigmoid activation function.
    
    The Sigmoid function maps input values to a range between 0 and 1, 
    making it suitable for binary classification tasks. It is defined by the 
    formula: f(x) = 1 / (1 + e^(-x)). However, it suffers from vanishing gradient 
    issues in deep networks due to its saturation regions near ±1.
    
    Formula: f(x) = \frac{1}{1 + e^{-x}}
    
    """
    
    __attr__ = "MML.nn_Activation_Sigmoid"
    
    def __init__(self, 
                 *,
                 module_name: str = "nn_Activation_Sigmoid", 
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        An Sigmoid activation function.

        Parameters:
            module_name: str, The name of the module instance.
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """
        
        super().__init__(module_name = module_name, backend = backend, dtype = dtype, device = device, autograd = autograd)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the Sigmoid activation function to the input tensor.

        This method computes the element-wise Sigmoid activation, which maps input values
        to the range (0, 1). The Sigmoid function is widely used in neural networks for
        binary classification tasks due to its smooth, differentiable nature. The output
        is saved for use during backward propagation to compute gradients.

        Args:
            x (Tensor): Input tensor of any shape. The Sigmoid operation is applied element-wise.

        Returns:
            Tensor: Output tensor after applying the Sigmoid activation, with the same shape as the input.

        Raises:
            ValueError: If the input `x` is not a valid MML.Tensor object.

        """
                
        # Type check, x must be an instance of Tensor
        if isinstance(x, Tensor) == False:
            raise ValueError(f"In performing forward(), input `x` must be in a MML `Tensor` format but you have {type(x)}")
        
        # Perform a sigmoid function on the input
        output = x.sigmoid()
        
        # Save output for backward
        self.__setattr__("output", output)

        return output
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a module during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            grad_output (Tensor): Gradient tensor resulting from the output of the layer, used as input for backpropagation.

        Returns:
            Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers.

        Raises:
            ValueError: If `grad_output` is not a valid MML.Tensor object..
        """
        
        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            return None
        
        # Type check, grad_output must be an instance of Tensor
        if isinstance(grad_output, Tensor) == False:
            raise ValueError(f"In performing backward(), `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")
        
        # grad = grad_output * sigmoid(x) * (1 - sigmoid(x))
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input
    
    def __repr__(self):
        return "nn_Activation_Sigmoid(Sigmoid Activation Function)."
    
    
# Alias for nn_Activation_Sigmoid
Sigmoid = nn_Activation_Sigmoid


# Implementation of Tanh Activation
class nn_Activation_Tanh(nn_Module):
    """
    Tanh activation function.
    
    The hyperbolic tangent (tanh) function maps input values to a range between -1 and 1,
    making it suitable for scenarios requiring symmetric output distribution. It is defined as:
    
    Formula: f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \tanh(x)
    
    """
    
    __attr__ = "MML.nn_Activation_Tanh"
    
    def __init__(self, 
                 *,
                 module_name: str = "nn_Activation_Sigmoid", 
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        An Sigmoid activation function.

        Parameters:
            module_name: str, The name of the module instance.
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """
        
        super().__init__(module_name = module_name, backend = backend, dtype = dtype, device = device, autograd = autograd)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the Tangent Hyperbolic activation function to the input tensor.

        This method computes the element-wise hyperbolic tangent (tanh) activation,
        which maps input values to the range (-1, 1). The tanh function is smooth and
        differentiable everywhere, making it suitable for neural network layers that
        require non-linear transformations. The output is saved for use in backward
        propagation to compute gradients during training.

        Args:
            x (Tensor): Input tensor of any shape. The tanh operation is applied element-wise.

        Returns:
            Tensor: Output tensor after applying the tanh activation, with the same shape as the input.

        Raises:
            ValueError: If the input `x` is not a valid MML.Tensor object.
        """
                
        # Type check, x must be an instance of Tensor
        if isinstance(x, Tensor) == False:
            raise ValueError(f"In performing forward(), input `x` must be in a MML `Tensor` format but you have {type(x)}")
        
        # Perform a tanh function on the input
        output = x.tanh()
        
        # Save output for backward
        self.__setattr__("output", output)

        return output
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a module during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            grad_output (Tensor): Gradient tensor resulting from the output of the layer, used as input for backpropagation.

        Returns:
            Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers.

        Raises:
            ValueError: If `grad_output` is not a valid MML.Tensor object..
        """
        
        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            return None
        
        # Type check, grad_output must be an instance of Tensor
        if isinstance(grad_output, Tensor) == False:
            raise ValueError(f"In performing backward(), `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")
        
        # grad = grad_output * (1 - tanh(x)^2)
        grad_input = grad_output * (1 - self.output ** 2)
        return grad_input
    
    def __repr__(self):
        return "nn_Activation_Tanh(Tanh Activation Function)."
   
    
# Alias for nn_Activation_Tanh
Tanh = nn_Activation_Tanh

   
# Implementation of Softmax Activation
class nn_Activation_Softmax(nn_Module):
    """
    Softmax activation function.
    
    The Softmax function converts raw scores (logits) into probabilities 
    that sum to 1, making it suitable for multi-class classification tasks. 
    It generalizes the sigmoid function to multiple classes by applying the 
    formula: f(x_i) = exp(x_i) / sum_j(exp(x_j)), where x_i is the input score 
    for class i. This ensures the output represents a probability distribution 
    over the classes.
    
    Formula: f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
    
    """
    
    __attr__ = "MML.nn_Activation_Softmax"
    
    def __init__(self, 
                 dim: int = 1,
                 *,
                 module_name: str = "nn_Activation_Sigmoid", 
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        An Sigmoid activation function.

        Parameters:
            dim: int, The dimension to apply softmax on. Defaults to 1.
            module_name: str, The name of the module instance.
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.softmax_dim: int, The dimension to apply softmax on.
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """
        
        super().__init__(module_name = module_name, backend = backend, dtype = dtype, device = device, autograd = autograd)
    
        # Record the softmax dimension as a non-Parameter attribute
        self.__setattr__("softmax_dim", dim)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the Softmax activation function to the input tensor.

        This method applies the softmax function along the specified axis (softmax_dim)
        to convert logits into probabilities. The output is a Tensor with the same shape
        as the input, but with values normalized to the range [0, 1] along the specified axis.
        This operation is commonly used in classification tasks to produce probability distributions.

        Args:
            x (Tensor): Input tensor containing logits (unnormalized log probabilities).
            softmax_dim (int): The axis along which to apply the softmax function. 
                              For example, for a batch of images, this could be the channel dimension.

        Returns:
            Tensor: Output tensor with probabilities computed via softmax along the specified axis.

        Raises:
            ValueError: If the input `x` is not a valid MML.Tensor object.

        """
                
        # Type check, x must be an instance of Tensor
        if isinstance(x, Tensor) == False:
            raise ValueError(f"In performing forward(), input `x` must be in a MML `Tensor` format but you have {type(x)}")
        
        # Perform a softmax function on the input
        output = x.softmax(axis = self.softmax_dim, keepdims = True)
        
        # Save output for backward
        self.__setattr__("output", output)

        return output
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a module during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            grad_output (Tensor): Gradient tensor resulting from the output of the layer, used as input for backpropagation.

        Returns:
            Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers.

        Raises:
            ValueError: If `grad_output` is not a valid MML.Tensor object..
        """
        
        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            return None
        
        # Type check, grad_output must be an instance of Tensor
        if isinstance(grad_output, Tensor) == False:
            raise ValueError(f"In performing backward(), `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")
        
        # Compute gradient w.r.t input using Jacobian: grad_input = y * (grad_out - (grad_out * y).sum_along_dim)
        grad_sum = (grad_output * self.output).sum(axis=self.softmax_dim, keepdims=True) 
        grad_input = self.output * (grad_output - grad_sum)
        return grad_input
    
    def __repr__(self):
        return "nn_Activation_Softmax(Softmax Activation Function)."
   
    
# Alias for nn_Activation_Softmax
Softmax = nn_Activation_Softmax
 

# Test case of Activations
if __name__ == "__main__":
    
    from nn import Dense
    from nn import Softmax
    from nn import Module, nn_Module
    
    class any_test(Module):
        
        def __init__(self):
            
            super().__init__(module_name="any_test")
            self.dense = Dense(4, 2, True)
            self.softmax = Softmax()
            self.sumover = Dense(2, 1, False)
        
        def forward(self, inputs):
            out = self.dense.forward(inputs)
            out = self.softmax.forward(out)
            out = self.sumover.forward(out)
            return out
    
    inputs = Tensor([[1,2,3,4.],[2,3,4,5]], backend="torch")
    difference = Tensor([[0.002312],[0.002341]], backend="torch")    
    
    # Test forward
    x = any_test()
    x.train()
    x.forward(inputs)

    # Test backward
    x.backward(difference)
    