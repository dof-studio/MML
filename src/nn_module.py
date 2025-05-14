# nn_module.py
#
# A Deep Neural Network Interface Module Class
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from typing import Any, Literal
    
from objtyp import Object
from tensor import Tensor

from nn_base import nn_Base
from nn_parameter import nn_Parameter

from copy import deepcopy


# A Deep Neural Network Interface Module Base Class
class nn_BaseModule(nn_Base):
    """
    A even base class for Neural Network Modules.
    """
    
    __attr__ = "MML.nn_BaseModule"    
    
    def __setattr__(self, name: str, value: Any):
        """
        Override setattr to register Parameters and Modules.
        
        Registers parameters (nn_Parameter) and modules (nn_Module) under the `._parameters` and `._modules` dictionaries 
        when appropriate. Delegates attribute assignment to the base class's `__setattr__` method for standard attributes.
        
        Args:
            name: str, The name of the attribute being set.
            value: Any, The value to assign to the attribute.
            
        Returns:
            None
        """
        # Register a parameter if is nn_Parameter
        if isinstance(value, nn_Parameter):
            self._parameters[name] = value
            
        # Register a submodule if nn_Module
        elif isinstance(value, nn_BaseModule):
            self._modules[name] = value
            
        # Copy if it is a Object based variable
        elif isinstance(value, Object):
            object.__setattr__(self, name, value)
        
        # In all cases, set the attribute normally as an attribute
        object.__setattr__(self, name, value)
    
    def __call__(self, *inputs):
        """
        Delegates to the forward method to perform the forward pass of the module.
        
        This method allows a Module instance to be called like a function, passing inputs to the forward() method.
        It delegates attribute assignment and functionality to the base class's `__setattr__` and `forward()` methods.
        
        Args:
            *inputs: list, Variable number of input tensors or values to pass to the forward method.
        
        Returns:
            Tensor: The output tensor resulting from the forward pass.
        
        Raises:
            NotImplementedError: If the forward() method is not implemented in the subclass.
        """
        # Allows Module instance to be called like a function to perform forward pass
        return self.forward(*inputs)
    
    def __init__(self, module_name: str = "Default_Module_Container"):
        """
        Initializes a base neural network module with basic structural components.

        This constructor sets up essential properties for a module, including its name,
        training mode flag, and containers for parameters and submodules. It follows a
        convention similar to PyTorch modules, where modules are organized hierarchically
        with parameter tracking and submodule management.

        Parameters:
            --------
            module_name: str, The name of the module instance. Defaults to "Default_Module_Container".

        Attributes:
            self.name: The name of the module instance, set dynamically via __setattr__.
            self.training: A flag indicating whether the module is in training mode. Defaults to True.
            self._parameter: A dictionary container for all parameters of this module.
            self._modules: A dictionary container for all submodule instances nested within this module.
        """
        
        # A default module at least contains:
        # 1. self.name, str, the name of this module instance
        # 2. self.training, bool, whether the module is in training or evaluation mode
        # 3. self.accumulate, bool, whether the module is accumulating gradients
        # 4. self._parameters, container, all parameters of THIS module
        # 5. self._modules, container, all instances of SUB modules
        
        # Module name, indicating the name of the module, a string
        self.__setattr__("name", module_name)
        
        # Training flag, indicating the model is training or not
        self.__setattr__("training", False)
        
        # Accumulating flag, indicating the model is accumulating gradients or not
        self.__setattr__("accumulate", False)
        # You can only set this to True by calling accumulate_grad before doing forward
        # to accumulate gradients when training.
        
        # Initialize internal containers for parameters
        self.__setattr__("_parameters", {})
        self.__setattr__("_modules", {})

    def forward(self, *inputs):
        """
        Override this method in subclasses to define forward pass.
        A forward pass is the way that the neural network passes the inputs
        through parameters and generates the output.
        
        Raises:
            NotImplementedError: You should implement your own forward pass. Calling the base will raise this error.
        """
        raise NotImplementedError("forward() is not implemented in the base module. You should define the architecture of your neural network manually.")

    def parameters(self):
        """
        Returns an iterator (or list) of all Parameters in this module, including those from child modules.

        This method recursively collects all `Parameter` instances from the current module and its submodules,
        following a pattern similar to PyTorch's `parameters()` method. It aggregates parameters from both direct
        parameters (`self._parameters`) and nested modules (`self._modules`).

        Returns:
            list: An iterator of all `Parameter` objects in this module and its submodules.
        """
        params = []
        
        # Own parameters
        for param in self._parameters.values():
            params.append(param)
            
        # Parameters of submodules
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        """
        Resets the gradients of all parameters in this module and its submodules to zero.

        This method iterates through all `Parameter` objects in the module (including those from child modules)
        and calls `zero_grad()` on each, effectively clearing the gradient buffers. It follows the same pattern
        as PyTorch's `zero_grad()` method for efficiency and consistency with standard neural network training workflows.

        Returns:
            None
        """
        for param in self.parameters():
            param.zero_grad()

    def accumulate_grad(self):
        """
        Set module to gradient accumulate mode. 

        This method sets the `accumulate` flag to True, ensures gradients are accumulated in backward passing instead of being set to 0.
        You should first call `train` to turn the module into training model. Otherwise it will raise a RuntimeError.

        Returns:
            None
            
        Raises:
            RuntimeError: if calling accumulate_grad in non-training mode.
        """
        # If called in non-training mode, raise RuntimeError
        if self.training == False:
            raise RuntimeError(f"Calling accumulate_grad() on Module {self.name} to accumulate gradients, but without turning training mode on. Please call .train() first.")
        
        # Set the status to accumulating
        self.accumulate = True
        
        # For other modules, set to accumulate mode
        for module in self._modules.values():
            module.accumulate_grad()
        
    def train(self):
        """
        Set module to training mode (affects dropout, and enables gradients).

        This method sets the `training` flag to True, ensures all direct parameters require gradients, and recursively applies 
        the training mode to all submodules. It is typically used at the beginning of a training loop to activate behaviors 
        specific to training, such as dropout or batch normalization.

        Returns:
            None
        """
        # Set the status to training
        self.training = True
        
        # For this module, set all parameters to requires_grad = True
        for param in self._parameters.values():
            param.requires_grad_(True)
        
        # For other modules, set to train modes
        for module in self._modules.values():
            module.train()

    def eval(self):
        """
        Set module to evaluation mode (disable dropout and gradients).
        
        This method sets the `training` flag to False, which disables behaviors specific to training (e.g., dropout). 
        It also explicitly sets all direct parameters to require gradients (`requires_grad=True`) and recursively applies 
        evaluation mode to all submodules.

        Returns:
            None
        """
        # Set the status to evalutating
        self.training = False
        
        # For this module, set all parameters to requires_grad = False
        for param in self._parameters.values():
            param.requires_grad_(False)
        
        # For other modules, set to evaluation mode
        for module in self._modules.values():
            module.eval()

    def to(self, device: str | None = None):
        """
        Moves all parameters and submodules to the specified device.

        This method relocates the module's parameters and nested submodules to the given device (e.g., 'cpu', 'cuda'). 
        If `device` is None, it uses the default device. This is essential for moving models between devices during training or inference.

        Args:
            device (str | None): The target device (e.g., "cpu", "cuda"). If None, the default device is used.

        Returns:
            self
        """
        # Set device directly
        if device == self.device:
            return self
        else:
            self.device = device
        
        # For this module, set every parameters to device
        for param in self._parameters.values():
            param.to(device)
       
        # For other modules, set every parameters to device
        for module in self._modules.values():
            module.to(device)
        return self

    def backward(self, grad_output: Tensor | None):
        """
        Backpropagate through the module (feedforward).
        
        If you need gradients with respect to internal states like GRU or LSTM states,
        please manually override this method to capture the gradients.
        By default, we only capture the 1st gradient which is gradient with respect to inputs.

        This method performs gradient propagation through the module's submodules in reverse order of their addition.
        ** By default, if the module contains submodules, propagate grad through them in reverse order.
        ** Leaf modules (layers) should override this to implement their own backward logic.
        ** Leaf modules (layers) should also be compatible to pytorch's autograd and manual calculation.

        Args:
            grad_output (Tensor): The gradient tensor resulting from the output of the module, used as input for backpropagation..

        Returns:
            Tensor: The propagated gradient after processing through all submodules.

        Raises:
            ValueError: If `grad_output` is not a valid MML.Tensor object.
        """
        
        # Note. This will be override by Layers to implement the actual logic of 
        #       gradient calculation and backpropagation.
        # For a normal non-leaf module, we assume it is a container containing NO
        #       parameter and just sub-modules. So we invoke backward of submodules.
        
        # If autograd, then the true backward is performed by LOSS function
        # Although it gets soemthing here into the module, it just for compatibility.
        # We don't need to processs it anymore.
        if self.autograd == True:
            return None
        
        # Type check, it must be a Tensor object if non-autograd mode
        if isinstance(grad_output, Tensor) == False:
            raise ValueError(f"Output gradient must be in a MML `Tensor` format but you have {type(grad_output)}")
            
        # Status check, if non-training, RuntimeErrror
        if self.training == False:
            raise RuntimeError(f"Backward can only be called in training mode but your current module {self.name} is not in the training mode.")
            
        # Propagate gradient through submodules in `reverse` order of addition
        for module in reversed(list(self._modules.values())):
            # For GNU or layers with hidden spaces, backward may
            # return more than 1 element, but the 1st is ensured to be 
            # grad wrt to inputs.
            # In the general feed-forward case, we only keep the 1st gradient.
            # If you need other terms, override this method to achive it.
            grad_output = module.backward(grad_output)
            if isinstance(grad_output, tuple):
                grad_output = grad_output[0]
            
        return grad_output
    
    def copy(self):
        """
        Create a deepcopy of the parameters and gradiets.
        """
        return deepcopy(self)
    
    def __repr__(self):
        return "nn_BaseModule(Deep Neural Network Base Module Class \nConstruct Your Own Network by Creating Children of this Module.)."


# A Deep Neural Network Interface Module Class
# Construct Your Own Network by Creating Children of this Module
class nn_Module(nn_BaseModule):
    
    """
    Base interface for all neural network modules, with standard keyword arguments.
    
    All neural network modules implemented by users should inherit this class
    and utilize the forward() to define the architecture of their networks.
    By interacting with an optimizer and loss function, you can update
    the weights stored in Parameters and train your neural network.
    """
    
    __attr__ = "MML.nn_Module"    

    def __init__(self, 
                 *,
                 module_name: str = "Default_Module_Container",
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        Initializes a neural network module interface with basic structural components.

        This constructor sets up essential properties for a module, including its name,
        training mode flag, and containers for parameters and submodules. It follows a
        convention similar to PyTorch modules, where modules are organized hierarchically
        with parameter tracking and submodule management.

        Parameters:
            --------
            module_name: str, The name of the module instance. Defaults to "Default_Module_Container".
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.name: The name of the module instance, set dynamically via __setattr__.
            self.training: A flag indicating whether the module is in training mode. Defaults to True.
            self._parameter: A dictionary container for all parameters of this module.
            self._modules: A dictionary container for all submodule instances nested within this module.
            self.backend: Literal["torch", "numpy"], The computational backend used by the layer.
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
        """
        
        # A default module at least contains:
        # 1. self.name, str, the name of this module instance
        # 2. self.training, bool, whether the module is in training or evaluation mode
        # 3. self.accumulate, bool, whether the module is accumulating gradients
        # 4. self._parameters, container, all parameters of THIS module
        # 5. self._modules, container, all instances of SUB modules
        # 6. self.backend, str, saying which backend Tensor will by default use
        # 7. self.dtype, type, saying the type the data will be stored
        # 8. self.device, str, saying the device where the data is stored on
        # 9. self.autograd, bool, whether the modules use pytorch autograd or not
        
        # Module name, internal containers, calling base init to initialize
        super().__init__(module_name = module_name)
        
        # Process the default types
        if backend not in ("numpy", "torch"):
            raise ValueError(f"In creating a module {module_name}, an unsupported backend is passed in. Use 'numpy' or 'torch' only.")
        if backend == "numpy":
            dtype = np.float32 if dtype is None else dtype
            device = "cpu" if device is None else device
        elif backend == "torch":
            dtype = torch.float32 if dtype is None else dtype
            device = "cpu" if device is None else device
        
        # Record the backend, dtype, device, autograd traits
        self.__setattr__("backend", backend)
        self.__setattr__("dtype", dtype)
        self.__setattr__("device", device)
        self.__setattr__("autograd", autograd)
        
    def __repr__(self):
        return f"nn_Module(name = {self.name})."


# Alias for nn_Module
Module = nn_Module


