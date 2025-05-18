# nn_layer_rnn.py
#
# An RNN and Stacked RNN Implementation
# From MML Library by Nathmath


import numpy as np
try:
    import torch
except ImportError:
    torch = None
    
from copy import deepcopy

from collections import deque
from typing import Any, List, Tuple, Literal
    
from .objtyp import Object
from .tensor import Tensor

from .nn_parameter import nn_Parameter
from .nn_module import nn_Module


# Implementation of a Vanilla RNN Cell using TBPTT
class nn_Layer_RNNCell(nn_Module):
    """
    RNN Cell Implementation
    
    This class implements a basic Recurrent Neural Network (RNN) cell, 
    designed to process sequential data by maintaining and updating a hidden state 
    at each time step. It contains weight matrices and bias vectors stored 
    in nn_Parameter containers, enabling the computation of forward passes through 
    sequences while supporting gradient-based learning via backward passes. 
        
    Note, TBPTT should be implemented externally with segmentation and carrying (manually reset_hidden).
    """

    __attr__ = "MML.nn_Layer_RNNCell"   

    def __init__(self, 
                 in_features: int = 1, 
                 hid_features: int = 1, 
                 has_bias: str = True,
                 init_scale: float = 0.1,
                 actv: Literal['tanh', 'relu'] = 'tanh', 
                 tbptt_steps: int | None = None,
                 return_sequences: bool = False, 
                 return_state: bool = False,
                 accumulate_hidden: bool = False,
                 gradient_clipping: float = 5.0,
                 *,
                 module_name: str = "nn_Layer_RNNCell", 
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        An RNN (Recurrent Neural Network) cell processes sequences by maintaining
            a hidden state that is updated at each time step.
            
        Structure: h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh}) \tag{1}
        
        Parameters:
            in_features: int, The number of input features for this layer. Defaults to 1.
            hid_features: int, The number of hidden features for this layer. Defaults to 1.
            has_bias: str, A flag indicating whether to include a bias term. Valid values are "True" or "False".
                    If set to "True", the layer includes an additive bias term (b). Defaults to "True".
            actv: str, activation type, must be in 'tanh' and 'relu'. Defaults to 'tanh'.
            tbptt_steps: int | None, the number of steps performing Truncated BPTT (TBPTT), 
                    if None, then use true BPTT. Defaults: None.
                    Note, TBPTT is not fully implemented. Please use BPTT instead.
            return_sequences: bool, should the layers return only the final hidden state, or the full sequence 
                    of hidden states as well. Defaults to False.
            return_state: bool, should the layers return output and final hidden state. Defaults to False.
            accumulate_hidden: bool, should the network accumulate hidden state (when processing segments) or clear it before next forward.
            gradient_clipping: float, The factor of performing gradient clipping to avoid gradient expolosion. Defaults to 5.0.
            module_name: str, The name of the module instance. Defaults to "nn_Layer_RNNCell".
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.in_features: int, The number of input features for this layer.
            self.hid_features: int, The number of output features for this layer.
            self.has_bias: str, A flag indicating whether a bias term is included ("True" or "False").
            self.init_scale: float, A floatting number indicating the maximum value of initial random weights.
            self.actv: Literal["tanh", "relu"], the activation type, either "tanh" or "relu".
            self.tbptt_steps: int | None, the number of TBPTT size, or left None.
            self.return_sequences: bool, indicator of whether returns full sequence of hidden or not.
            self.return_state: bool, indicator of whether returns final state or not.
            self.accumulate_hidden: bool, indicator of whether accumulate states or reinitialize when next forward.
            self.gradient_clipping: float, The threshold for gradient clipping.
            self.backend: Literal["torch", "numpy"], The computational backend used by the layer.
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight_ih: nn_Parameter, input weights
            self._parameters.bias_ih: nn_Parameter | None, input bias
            self._parameters.weight_hh: nn_Parameter, hidden weights
            self._parameters.bias_hh: nn_Parameter | None, hidden bias
            self._cache: dict of list of tensors, for backward propagation
        """
        
        super().__init__(module_name = module_name, backend = backend, dtype = dtype, device = device, autograd = autograd)
        
        # If actv is not in 'tanh' or 'relu', raise ValueError
        if actv not in {'tanh', 'relu'}:
            raise ValueError(f"In initializing an RNNCell, you have to ensure `actv` in 'relu' or 'tanh', but you have {actv}")
        
        # Record shapes etc
        self.__setattr__("in_features", in_features)
        self.__setattr__("hid_features", hid_features)
        self.__setattr__("has_bias", has_bias)
        self.__setattr__("init_scale", init_scale)
        self.__setattr__("actv", actv)
        self.__setattr__("tbptt_steps", tbptt_steps)
        self.__setattr__("return_sequences", return_sequences)
        self.__setattr__("return_state", return_state)
        self.__setattr__("accumulate_hidden", accumulate_hidden)
        self.__setattr__("gradient_clipping", gradient_clipping)
        
        # Cache: to store values needed for backward
        self.__setattr__("_cache", {})
        
        # Initialize weight and bias parameters
        
        # weight_ih: hidden_size * input_size
        # weight_hh: hidden_size * hidden_size
        self.__setattr__("weight_ih", nn_Parameter(
            Tensor.rand([hid_features, in_features], backend=backend, dtype=dtype, device=device) * init_scale,
            requires_grad = True,
            dtype = None,
            device = None,
            autograd = autograd)
            )
        self.__setattr__("weight_hh", nn_Parameter(
            Tensor.rand([hid_features, hid_features], backend=backend, dtype=dtype, device=device) * init_scale,
            requires_grad = True,
            dtype = None,
            device = None,
            autograd = autograd)
            )
        
        if has_bias == True:
            # If uses bias, then set the bias
            self.__setattr__("bias_ih", nn_Parameter(
                Tensor.zeros([hid_features], backend=backend, dtype=dtype, device=device),
                requires_grad = True,
                dtype = None,
                device = None,
                autograd = autograd)
                )
            self.__setattr__("bias_hh", nn_Parameter(
                Tensor.zeros([hid_features], backend=backend, dtype=dtype, device=device),
                requires_grad = True,
                dtype = None,
                device = None,
                autograd = autograd)
                )

    def reset_hidden(self) -> None:
        """
        Reset the stored hidden state if done TBPTT.
        If invoked, when calling forward() next time, hiddden state will be starting from 0s.
        """
        if self._cache.get('hidden_final', None) is not None:
            self._cache.pop('hidden_final')
            
    def clear_cache(self) -> None:
        """
        Clear the cache dict to release Tensors and avoid issues.
        If invoked, cache will be an empty dict. Do not invoke before applying backward().
        """
        self._cache = {}

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor| Tuple[Tensor, Tensor]:
        """
        Compute the layer output: h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh}) \tag{1} and return it.
        You can safely choose NOT to return hidden states since we have memorized them if you do not pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)) to be transformed by the layer.
            h0 (Tensor | None): initial hidden state, Tensor of shape (batch, hidden_size) or left None.
                If left None, the net will automatically use the previous final hidden state stored in the cache if selected..

        Returns:
            Tensor: Output tensor of shape (batch_size, hid_features) after applying the RNN transformation.
            or 
            A tuple of Tensors of 2 having outputs and the hidden state.

        Raises:
            ValueError: If the input `x` is not a valid MML.Tensor object.
            ValueError: If the input `h0` is given but not equal to the (batch_size, hid_features).

        Attributes:
            self.input (Tensor): The input tensor `x` saved for use in backward propagation.
        """
        
        # Type check, x must be an instance of Tensor
        if isinstance(x, Tensor) == False:
            raise ValueError(f"In performing forward(), input `x` must be in a MML `Tensor` format but you have {type(x)}")
        
        # If h0 is given, it must have (batch_size, hid_features) size
        if h0 is not None:
            if isinstance(h0, Tensor) == False:
                raise ValueError(f"In performing forward(), input `h0` must be in a MML `Tensor` format but you have {type(h0)}")
            if h0.shape != (x.shape[0], self.hid_features):
                raise ValueError(f"In performing forward(), if given an input `h0`, it must equal to (batch_size, hidden_size) but you have {h0.shape}")
        # Else if it has been stored previously, reuse it
        else:
            if self._cache.get('hidden_final', None) is not None and self.accumulate_hidden == True:
                h0 = self._cache['hidden_final']
        
        # Save input for backward
        self.__setattr__("input", x)
        
        # Rrcord the critical shapes
        batch_size, seq_len, *_ = x.shape
        
        # Initialize hidden state
        if h0 is None:
            h = Tensor.zeros((batch_size, self.hid_features), backend=self.backend, dtype=self.dtype, device=self.device)
        else:
            h = h0
            
        # Cache lists for backward (to compute gradients)
        window = self.tbptt_steps if (self.tbptt_steps is not None and self.tbptt_steps > 0) else seq_len
        x_cache  = deque(maxlen=window)
        h_lin_cache = deque(maxlen=window)
        h_post_cache = deque(maxlen=window)
            
        # Prepare output list if return_sequences
        outputs = [] if self.return_sequences == True else None
        
        # Forward propagate through time
        for t in range(seq_len):
            
            # Crcord input at time t, shape (batch, input_size)
            xt = x[:, t, ...]  
            x_cache.append(xt)

            # Compute h_lin = W_ih * x_t + W_hh * h_{t-1} + b_ih + b_hh
            h_lin = (xt @ self._parameters["weight_ih"].data.transpose()) + (h @ self._parameters["weight_hh"].data.transpose())
            if self.has_bias:
                h_lin = h_lin + self._parameters["bias_ih"].data
                h_lin = h_lin + self._parameters["bias_hh"].data

            # Apply nonlinearity
            if self.actv == 'relu':
                h_new = h_lin.relu()
            else:  # default 'tanh'
                h_new = h_lin.tanh()

            # Store pre-activation and post-activation hidden for backprop
            h_lin_cache.append(h_lin)
            h_post_cache.append(h_new)

            # Append to outputs if needed
            if self.return_sequences == True:
                outputs.append(deepcopy(h_new))

            # Prepare next step
            h = h_new  # update hidden state

            # TBPTT: Truncated Back Propagation Through Time
            # if we've reached tbptt_steps, detach hidden state
            if self.tbptt_steps is not None and self.tbptt_steps > 0:
                
                # Detach at the boundary of each segment
                if (t + 1) % self.tbptt_steps == 0:
                    # Detach the hidden state to prevent backprop beyond this point
                    # It is only used for autograd purposes
                    h = h.detach()  
                    
        # Save cache for backward
        if self.training == True:
            self._cache['x'] = x_cache
            self._cache['h_lin'] = h_lin_cache
            self._cache['h'] = h_post_cache
            self._cache['seq_len'] = seq_len
            self._cache['batch_size'] = batch_size
            self._cache['hidden_final'] = h  
        # h: final hidden state after last step (stored for continuation reuse)

        # Prepare output data
        if self.return_sequences == True:
            # Stack outputs list to tensor: (batch, seq_len, hidden)
            outputs_tensor = outputs[0].stack(*outputs[1:], axis=1)
        else:
            # Last output is the last hidden state
            outputs_tensor = h

        if self.return_state == True:
            # Return output and final hidden state
            return outputs_tensor, h
        else:
            # Return just the output tensor
            return outputs_tensor
    
    def backward(self, grad_output: Tensor, grad_h: Tensor | None = None) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a dense layer during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            grad_output (Tensor): Gradient tensor resulting from the output of the layer, used as input for backpropagation.
                     - If return_sequences=True, shape is (batch, seq_len, hidden_size)
                     - If return_sequences=False, shape is (batch, hidden_size) for last output.
            grad_h (Tensor | None): Gradient of loss w.r.t. the final hidden state.
                     - This is usually None unless the RNN's final state is fed into another layer.

        Returns:
            Tuple of 
                (Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers, shape (batch, seq_len, input_size) tensor.
                 Tensor: Gradient with respect to the final hidden state, ∂L/∂h0.
                 )
        Raises:
            ValueError: If `grad_output` is not a valid MML.Tensor object.
            ValueError: If `grad_h` is given but not a valid MML.Tensor object.
        """
        
        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            return None
        
        # Type check, grad_output must be an instance of Tensor
        if isinstance(grad_output, Tensor) == False:
            raise ValueError(f"In performing backward(), `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")
        
        # Type check, if given, grad_h must be an instance of Tensor
        if grad_h is not None:
            if isinstance(grad_h, Tensor) == False:
                raise ValueError(f"In performing backward(), if given `grad_h`, it must be in a MML `Tensor` format but you have {type(grad_h)}")
        
        # Retrieve cached values
        x_cache = self._cache.get('x')
        h_lin_cache = self._cache.get('h_lin')
        h_post_cache = self._cache.get('h')
        seq_len = self._cache.get('seq_len')
        batch_size = self._cache.get('batch_size')
        
        # storage for dL/dx_t
        grad_x_all = Tensor.zeros((batch_size, seq_len, self.in_features), backend=self.backend, dtype=self.dtype, device=self.device)

        # Initialize gradient for hidden state (dL/dh) at step t (post-activation)
        # Start from final output gradient.
        if self.return_sequences == True:
            # If the output was the full sequence, grad_output is a tensor of shape (batch, seq_len, hidden)
            # We will accumulate gradient from each time step's output.
            # Start with zero grad for final hidden; contributions will come from grad_output at each step.
            grad_h_t = Tensor.zeros((batch_size, self.hid_features), backend=self.backend, dtype=self.dtype, device=self.device)
        else:
            # If output was last hidden only, grad_output is dL/dh_final
            grad_h_t = grad_output.clone()
            grad_output = None  # this is not used for per-step outputs in this case

        # If an external grad_h is provided, add it
        if grad_h is not None:
            grad_h_t += grad_h
        
        # maximum number of steps to walk back
        max_steps = self.tbptt_steps if (self.tbptt_steps and self.tbptt_steps > 0) else seq_len
        
        # Truncated Backpropagate Through Time (TBPTT)
        # We will iterate backwards through each time step.
        # If return_sequences, include grad from that step's output as well.
        steps_done = 0
        for t in range(seq_len - 1, -1, -1):
            
            if steps_done >= max_steps:
                break
            steps_done += 1
            
            # If return_sequences, accumulate gradient from output at time t
            if self.return_sequences == True:
                # grad_output[:, t, ...] is gradient of loss w.r.t output at time t
                grad_h_t += grad_output[:, t, ...]

            # h_post_cache[t] is h_t (post-nonlinearity), h_lin_cache[t] is pre-nonlinearity
            # grad_h_t is dL/dh_t (gradient wrt post-activation hidden state at time t)
            xt = x_cache[t]
            h_lin = h_lin_cache[t]
            h_post = h_post_cache[t]

            # Apply activation derivative
            if self.actv == 'relu':
                # derivative of ReLU: 1 for h_lin > 0, else 0
                grad_h_lin = grad_h_t * Tensor.where_as(h_lin.data > 0, 1.0, 0.0, backend=self.backend, dtype=self.dtype, device=self.device)
            else:
                # derivative of tanh: (1 - tanh^2) = 1 - h_post^2
                grad_h_lin = grad_h_t * (1.0 - h_lin.tanh() ** 2)

            # Now grad_h_lin is dL/d(h_lin) which is dL/d(W_ih x + W_hh h_prev + b)
            self._parameters["weight_ih"].grad += grad_h_lin.transpose() @ xt
            
            # Grad w.rt bias_ih: just sum grad_h_lin over batch
            if self.has_bias == True:
                self._parameters["bias_ih"].grad += grad_h_lin.sum(axis=0)
            
            # Grad w.rt weight_hh: uses previous hidden state (post-activation) at t-1 (or h0 for t=0)
            if t == 0:
                # previous hidden is either provided h0 or zero (we treat it as constant, so no gradient beyond it)
                h_prev = Tensor.zeros_like(h_post, backend=self.backend, dtype=self.dtype, device=self.device)  
            else:
                h_prev = h_post_cache[t-1] 
            
            # Now grad_h_lin is dL/d(h_lin) which is dL/d(W_hh x + W_hh h_prev + b)
            self._parameters["weight_hh"].grad += grad_h_lin.transpose() @ h_prev
            
            # Grad w.rt bias_hh: just sum grad_h_lin over batch
            if self.has_bias == True:
                self._parameters["bias_hh"].grad += grad_h_lin.sum(axis=0)

            # Gradients w.r.t. inputs and h_{t-1} grad_x_t = grad_h_lin * W_ih (matrix multiply)
            grad_x_all[:, t, ...] = grad_h_lin @ self._parameters["weight_ih"].data

            # grad_h_prev (dL/dh_{t-1}, post-activation) = grad_h_lin * W_hh
            grad_h_prev = grad_h_lin @ self._parameters["weight_hh"].data
            
            # Set grad_h_t for next iteration (previous time step)
            grad_h_t = grad_h_prev
            
        # After accumulating all gradients
        def _clip_gradients_inplace(params: nn_Parameter, max_norm: float) -> None:
            
            # Gather all gradient Tensors from nn_Parameter
            grads = [p.grad for p in params if p is not None and p.grad is not None]
            # List
            
            # Compute squared norms of each gradient and sum them
            total_norm_sq = sum(((g ** 2).sum()) for g in grads)
            # Tensor
            
            # Take the square root for the global norm
            total_norm = total_norm_sq ** 0.5
            
            # If outside threshold, scale each gradient in place
            if total_norm.to_list() > max_norm:
                scale = max_norm / (total_norm + 1e-12)
                for g in grads:
                    g *= scale
                    
            return
                    
        # If self.gradient_clipping is not None, then perform gradient clipping
        if self.gradient_clipping is not None and self.gradient_clipping > 0:
            parameters = [self._parameters["weight_ih"], 
                          self._parameters["weight_hh"], 
                          self._parameters["bias_ih"], 
                          self._parameters["bias_hh"]]
            _clip_gradients_inplace(parameters, self.gradient_clipping)
        
        return grad_x_all, grad_h_t
    
    def __repr__(self):
        return f"nn_Layer_RNNCell(shape: ({self.in_features}, {self.hid_features}) with{'out' if self.has_bias == False else ''} bias)."
    
    
# Implementation of a Stacked RNN Layer using TBPTT  
class nn_Layer_StackedRNN(nn_Module):
    
    """
    Stacked RNN Implementation
    
    This class implements a stacked Recurrent Neural Network (RNN) layer, 
    designed to process sequential data by maintaining and updating a hidden state 
    at each time step. It contains weight matrices and bias vectors stored 
    in nn_Parameter containers, enabling the computation of forward passes through 
    sequences while supporting gradient-based learning via backward passes. 
    
    Note, TBPTT should be implemented externally with segmentation and carrying (manually reset_hidden).
    """

    __attr__ = "MML.nn_Layer_StackedRNN"   

    def __init__(self, 
                 in_features: int = 1, 
                 hid_features: int = 1, 
                 num_layers: int = 1,
                 has_bias: str = True,
                 init_scale: float = 0.1,
                 actv: Literal['tanh', 'relu'] = 'tanh', 
                 tbptt_steps: int | None = None,
                 return_sequences: bool = False, 
                 return_state: bool = False,
                 accumulate_hidden: bool = False,
                 gradient_clipping: float = 5.0,
                 *,
                 module_name: str = "nn_Layer_StackedRNN", 
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        A Stacked RNN (Recurrent Neural Network) cell processes sequences by maintaining
            a hidden state that is updated at each time step.
            
        Structure: h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh}) \tag{1}
        
        Parameters:
            in_features: int, The number of input features for this layer. Defaults to 1.
            hid_features: int, The number of hidden features for this layer. Defaults to 1.
            num_layers: int, The number of RNN cells stacked together (h(n-1) becomes x in stack n).
            has_bias: str, A flag indicating whether to include a bias term. Valid values are "True" or "False".
                    If set to "True", the layer includes an additive bias term (b). Defaults to "True".
            actv: str, activation type, must be in 'tanh' and 'relu'. Defaults to 'tanh'.
            tbptt_steps: int | None, the number of steps performing Truncated BPTT (TBPTT), 
                    if None, then use true BPTT. Defaults: None.
                    Note, TBPTT is not fully implemented. Please use BPTT instead.
            return_sequences: bool, should the layers return only the final hidden state, or the full sequence 
                    of hidden states as well. Defaults to False.
            return_state: bool, should the layers return output and final hidden state. Defaults to False.
            accumulate_hidden: bool, should the network accumulate hidden state (when processing segments) or clear it before next forward.
            gradient_clipping: float, The factor of performing gradient clipping to avoid gradient expolosion. Defaults to 5.0.
            module_name: str, The name of the module instance. Defaults to "nn_Layer_StackedRNN".
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.in_features: int, The number of input features for this layer.
            self.hid_features: int, The number of output features for this layer.
            self.num_layers: int, The number of stacked RNN cells for this layer.
            self.has_bias: str, A flag indicating whether a bias term is included ("True" or "False").
            self.init_scale: float, A floatting number indicating the maximum value of initial random weights.
            self.actv: Literal["tanh", "relu"], the activation type, either "tanh" or "relu".
            self.tbptt_steps: int | None, the number of TBPTT size, or left None.
            self.return_sequences: bool, indicator of whether returns full sequence of hidden or not.
            self.return_state: bool, indicator of whether returns final state or not.
            self.accumulate_hidden: bool, indicator of whether accumulate states or reinitialize when next forward.
            self.gradient_clipping: float, The threshold for gradient clipping.
            self.backend: Literal["torch", "numpy"], The computational backend used by the layer.
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._modules[...]: dict, a dict of stacked RNN cells sequentially.
        """
        
        super().__init__(module_name = module_name, backend = backend, dtype = dtype, device = device, autograd = autograd)
        
        # If actv is not in 'tanh' or 'relu', raise ValueError
        if actv not in {'tanh', 'relu'}:
            raise ValueError(f"In initializing a Stacked RNN, you have to ensure `actv` in 'relu' or 'tanh', but you have {actv}")
        
        # Record shapes etc
        self.__setattr__("in_features", in_features)
        self.__setattr__("hid_features", hid_features)
        self.__setattr__("num_layers", num_layers)
        self.__setattr__("has_bias", has_bias)
        self.__setattr__("init_scale", init_scale)
        self.__setattr__("actv", actv)
        self.__setattr__("tbptt_steps", tbptt_steps)
        self.__setattr__("return_sequences", return_sequences)
        self.__setattr__("return_state", return_state)
        self.__setattr__("accumulate_hidden", accumulate_hidden)
        self.__setattr__("gradient_clipping", gradient_clipping)
        
        # Initialize stacked RNN components, each returning full sequence + its final state
        for i in range(num_layers):
            self.__setattr__("RNN_Layer_"+str(i), 
                             nn_Layer_RNNCell(
                                 in_features=in_features if i == 0 else hid_features,
                                 hid_features=hid_features,
                                 has_bias=has_bias,
                                 init_scale=init_scale,
                                 actv=actv,
                                 tbptt_steps=tbptt_steps,
                                 return_sequences=True,
                                 return_state=True,
                                 accumulate_hidden=accumulate_hidden,
                                 gradient_clipping=gradient_clipping,
                                 module_name=module_name+"_"+"RNN_Layer_"+str(i),
                                 backend=backend,
                                 dtype=dtype,
                                 device=device,
                                 autograd=autograd,
                                 **kwargs
                                 )
                             )

    def reset_hidden(self) -> None:
        """
        Reset the stored hidden state if done TBPTT.
        If invoked, when calling forward() next time, hiddden state will be starting from 0s.
        """
        for k in self._modules.keys():
            self._modules[k].reset_hidden()

    def clear_cache(self) -> None:
        """
        Clear the cache dict to release Tensors and avoid issues.
        If invoked, cache will be an empty dict. Do not invoke before applying backward().
        """
        for k in self._modules.keys():
            self._modules[k].clear_cache()

    def forward(self, x: Tensor, h0: List[Tensor] | None = None) -> Tensor| Tuple[Tensor, List[Tensor]]:
        """
        Compute the layer output: h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh}) \tag{1} and return it.
        You can safely choose NOT to return hidden states since we have memorized them if you do not pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)) to be transformed by the layer.
            h0 (List[Tensor] | None): initial hidden state, List of Tensors of shape (batch, hidden_size) or left None.

        Returns:
            Tensor: Output tensor of shape (batch_size, hid_features) after applying the RNN transformation.
            or 
            A tuple of 2 having outputs and the List of Tensors containing hidden state.

        Raises:
            ValueError: If the input `x` is not a valid MML.Tensor object.
            ValueError: If the input `h0` is given but not a list of Tensors.

        """
        # Type check, x must be an instance of Tensor
        if isinstance(x, Tensor) == False:
            raise ValueError(f"In performing forward(), input `x` must be in a MML `Tensor` format but you have {type(x)}")
        
        # If h0 is given, it must have (batch_size, hid_features) size
        if h0 is not None:
            if isinstance(h0, list) == False:
                raise ValueError(f"In performing forward(), input `h0` must be in a List of MML `Tensor` format but you have {type(h0)}")
            if isinstance(h0[0], Tensor) == False:
                raise ValueError(f"In performing forward(), if given an input `h0`, it must be a list of MML `Tensor`s but you have {type(h0[0])}")
            
        # Clear cache to release spaces
        self.clear_cache()
            
        # Prepare output and final states
        output = x
        final_states = []

        # Perform RNN Layer by layer
        for idx, key in enumerate(self._modules.keys()):
            h = h0[idx] if (h0 is not None) else None
            # Perform a forward pass for each layer sequentially
            output, h_n = self._modules[key].forward(output, h)
            final_states.append(h_n)
            
        # if only last step is wanted, slice it off the sequence
        if self.return_sequences == False:
            # (batch, hidden_size)
            output = output[:, -1, ...] 
            
        # If return state, then return a list of states
        if self.return_state == True:
            return output, final_states
        # Otherwise, only the output
        else:
            return output
        
    def backward(self, grad_output: Tensor, grad_h: List[Tensor] | None = None) -> Tensor | Tuple[Tensor, List[Tensor]]:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a dense layer during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            grad_output (Tensor): Gradient tensor resulting from the output of the layer, used as input for backpropagation.
                     - If return_sequences=True, shape is (batch, seq_len, hidden_size)
                     - If return_sequences=False, shape is (batch, hidden_size) for last output.
            grad_h (List[Tensor] | None): Gradient of loss w.r.t. the final hidden state.
                     - This is usually None unless the RNN's final state is fed into another layer.
                     - If used, it is List of Tensors with shape (batch, hidden_size)
        Returns:
            Tuple of 
                (Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers, shape (batch, seq_len, input_size) tensor.
                 List[Tensor]: Gradients with respect to each final hidden state, List[∂L/∂h0] (batch, hidden).
                 )
        Raises:
            ValueError: If `grad_output` is not a valid MML.Tensor object.
            ValueError: If `grad_h` is given but not a valid list of MML.Tensor object.
        """
        
        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            return None
        
        # Type check, grad_output must be an instance of Tensor
        if isinstance(grad_output, Tensor) == False:
            raise ValueError(f"In performing backward(), `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")
        
        # Type check, if given, grad_h must be an instance of Tensor
        if grad_h is not None:
            if isinstance(grad_h, list) == False:
                raise ValueError(f"In performing backward(), if given `grad_h`, it must be in a List of MML `Tensor`s format but you have {type(grad_h)} with each {type(grad_h[0])}")
            if isinstance(grad_h[0], Tensor) == False:
                raise ValueError(f"In performing backward(), if given `grad_h`, it must be in a List of MML `Tensor`s format but you have {type(grad_h)} with each {type(grad_h[0])}")
        
        # prepare per-layer state gradients
        if grad_h is None:
            grad_h = [None] * self.num_layers
            
        # grad flowing into top layer's output
        next_grad_out = grad_output
        grad_h0_list = [None] * self.num_layers

        # walk backward through layers
        keys = list(self._modules.keys())
        for i in reversed(range(self.num_layers)):
            layer     = self._modules[keys[i]]
            grad_h_n  = grad_h[i]
            grad_out  = next_grad_out

            # if only last-step output was used externally, expand it
            if self.return_sequences == False and i == self.num_layers - 1:
                seq_len = layer._cache["seq_len"]
                batch_size = layer._cache["batch_size"]
                hidden = layer.hid_features
                full = Tensor.zeros((batch_size, seq_len, hidden), backend=self.backend, dtype=self.dtype, device=self.device)
                full[:, -1, ...] = grad_out
                grad_out = full

            # call sub-layer backward (must return two)
            grad_x, grad_h0 = layer.backward(grad_out, grad_h_n)

            grad_h0_list[i] = grad_h0
            next_grad_out = grad_x

        # next_grad_out is gradient w.rt the very input x
        # grad_h0_list is gradient w.rt the hidden state of each layer
        return next_grad_out, grad_h0_list        

    def __repr__(self):
        return f"nn_Layer_StackedRNN(shape: ({self.in_features}, {self.hid_features}), num_layers: {self.num_layers} with{'out' if self.has_bias == False else ''} bias)."
    

# Alias for nn_Layer_StackedRNN
RNN = nn_Layer_StackedRNN
StackedRNN = nn_Layer_StackedRNN


# Test case for Stacked RNN
if __name__ == "__main__":
    
    from nn import Tensor
    from nn import Dense
    from nn import StackedRNN
    from nn import Sigmoid, ReLU
    from nn import Softmax
    from nn import Module, nn_Module
    from nn import RMSE, MSE, MultiCrossEntropy
    from nn import Adam
    from nn import Evaluator
    
    batch_size   = 32768
    seq_len      = 12
    input_size   = 8
    hidden_size  = 32
    output_size  = 1
    
    backend = "torch"
    device = "cuda"

    # inputs: (batch, seq_len, input_size)
    X = Tensor(torch.randn(batch_size, seq_len, input_size), backend=backend, device=device)
    # targets: (batch, output_size)
    y = (X.softmax(axis=2) - 0.5).sum(axis=1, keepdims=True)[...,0]
    y = y.reshape([-1, 1])
    y += 4 + y.to_rands() * 0.05
    
    class test_rnn(nn_Module):
        def __init__(self,
                     input_size, hidden_size, 
                     num_layers=3, tbptt_steps=None, **kwargs):
            
            super().__init__(**kwargs)
            
            # register the stacked RNN as a submodule
            self.stacked = StackedRNN(
                input_size, hidden_size,
                num_layers=num_layers,
                tbptt_steps=tbptt_steps,
                return_sequences=False,
                return_state=False,  # State is memorized
                **kwargs
            )
            # register the final dense head
            self.head = Dense(hidden_size, 1, **kwargs)
    
        def forward(self, x):
            out = self.stacked(x)
            out = self.head(out)
            return out
        
    x = test_rnn(input_size, hidden_size, backend = backend, device = device)
    
    x.train()
    
    crit = RMSE(backend=backend, device=device)
    optm = Adam(x.parameters(), lr=1e-4)
    
    for i in range(1000):
        out = x.forward(X)
        
        # Calculate loss
        loss = crit(out, y);
        print(loss.to_list())
        lossgrad = crit.backward()
    
        # Test backward
        x.backward(lossgrad)
        
        # Apply sgd
        optm.step()
        
        # Apply zero grad
        x.zero_grad()
        