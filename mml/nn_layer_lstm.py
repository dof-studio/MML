# nn_layer_lstm.py
#
# A LSTM and Stacked LSTM Implementation
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


# Implementation of a Vanilla Long Short Term Memory using TBPTT
class nn_Layer_LSTMCell(nn_Module):
    """
    LSTM Cell Implementation
    
    This class implements a Long Short-Term Memory (LSTM) cell, designed to process
    sequential data by maintaining and updating both hidden and cell states at each time step. 
    It contains multiple weight matrices (input, forget, cell, output gates) and corresponding 
    bias vectors stored in nn_Parameter containers, enabling the computation of forward 
    passes through sequences while supporting gradient-based learning via backward passes. 
    The LSTM architecture introduces gate mechanisms to control information flow, allowing
    for better capture of long-term dependencies compared to standard RNNs.
        
    Note, TBPTT should be implemented externally with segmentation and carrying (manually reset_hidden).
    """

    __attr__ = "MML.nn_Layer_LSTMCell"   

    def __init__(self, 
                 in_features: int = 1, 
                 hid_features: int = 1, 
                 has_bias: str = True,
                 init_scale: float = 0.1,
                 tbptt_steps: int | None = None,
                 return_sequences: bool = False, 
                 return_state: bool = False,
                 accumulate_hidden: bool = False,
                 gradient_clipping: float = 5.0,
                 *,
                 module_name: str = "nn_Layer_LSTMCell", 
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        An LSTM (Long Short Term Machine) cell processes sequences by maintaining
            a hidden state that is updated at each time step.
            
        Structure: 
            - **Input gate**: Controls how much new information is added to the cell state.  
              $$ i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) $$  
            - **Forget gate**: Determines what information to discard from the cell state.  
              $$ f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) $$  
            - **Cell gate (candidate)**: Computes a candidate value for the cell state.  
              $$ \tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c) $$  
            - **Output gate**: Regulates what part of the cell state is output as the hidden state.  
              $$ o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) $$  
        
        Parameters:
            in_features: int, The number of input features for this layer. Defaults to 1.
            hid_features: int, The number of hidden features for this layer. Defaults to 1.
            has_bias: str, A flag indicating whether to include a bias term. Valid values are "True" or "False".
                    If set to "True", the layer includes an additive bias term (b). Defaults to "True".
            tbptt_steps: int | None, the number of steps performing Truncated BPTT (TBPTT), 
                    if None, then use true BPTT. Defaults: None.
                    Note, TBPTT is not fully implemented. Please use BPTT instead.
            return_sequences: bool, should the layers return only the final hidden state, or the full sequence 
                    of hidden states as well. Defaults to False.
            return_state: bool, should the layers return output and final hidden state. Defaults to False.
            accumulate_hidden: bool, should the network accumulate hidden state (when processing segments) or clear it before next forward.
            gradient_clipping: float, The factor of performing gradient clipping to avoid gradient expolosion. Defaults to 5.0.
            module_name: str, The name of the module instance. Defaults to "nn_Layer_LSTMCell".
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
        
        # Record shapes etc
        self.__setattr__("in_features", in_features)
        self.__setattr__("hid_features", hid_features)
        self.__setattr__("has_bias", has_bias)
        self.__setattr__("init_scale", init_scale)
        self.__setattr__("tbptt_steps", tbptt_steps)
        self.__setattr__("return_sequences", return_sequences)
        self.__setattr__("return_state", return_state)
        self.__setattr__("accumulate_hidden", accumulate_hidden)
        self.__setattr__("gradient_clipping", gradient_clipping)
        
        # Cache: to store values needed for backward
        self.__setattr__("_cache", {})
        
        # Initialize weight and bias parameters
        
        # LSTM weight parameters (following PyTorch's gating order: i, f, g, o)
        # weight_ih: combines weights for input->[i, f, g, o] (shape: 4H x input_size)
        # weight_hh: combines weights for hidden->[i, f, g, o] (shape: 4H x hidden_size)
        # bias_ih, bias_hh: biases for the four gates (shape: 4H each)
        self.__setattr__("weight_ih", nn_Parameter(
            Tensor.rand([4 * hid_features, in_features], backend=backend, dtype=dtype, device=device) * init_scale,
            requires_grad = True,
            dtype = None,
            device = None,
            autograd = autograd)
            )
        self.__setattr__("weight_hh", nn_Parameter(
            Tensor.rand([4 * hid_features, hid_features], backend=backend, dtype=dtype, device=device) * init_scale,
            requires_grad = True,
            dtype = None,
            device = None,
            autograd = autograd)
            )
        
        if has_bias == True:
            # If uses bias, then set the bias
            self.__setattr__("bias_ih", nn_Parameter(
                Tensor.zeros([4 * hid_features], backend=backend, dtype=dtype, device=device),
                requires_grad = True,
                dtype = None,
                device = None,
                autograd = autograd)
                )
            self.__setattr__("bias_hh", nn_Parameter(
                Tensor.zeros([4 * hid_features], backend=backend, dtype=dtype, device=device),
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
        if self._cache.get('h0_final', None) is not None:
            self._cache.pop('h0_final')
        if self._cache.get('c0_final', None) is not None:
            self._cache.pop('c0_final')

    def clear_cache(self) -> None:
        """
        Clear the cache dict to release Tensors and avoid issues.
        If invoked, cache will be an empty dict. Do not invoke before applying backward().
        """
        self._cache = {}

    def forward(self, x: Tensor, h0: Tensor | None = None, c0: Tensor | None = None) -> Tensor| Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the layer output:
            - **Input gate**: Controls how much new information is added to the cell state.  
              $$ i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) $$  
            - **Forget gate**: Determines what information to discard from the cell state.  
              $$ f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) $$  
            - **Cell gate (candidate)**: Computes a candidate value for the cell state.  
              $$ \tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c) $$  
            - **Output gate**: Regulates what part of the cell state is output as the hidden state.  
              $$ o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) $$  
        You can safely choose NOT to return hidden states since we have memorized them if you do not pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)) to be transformed by the layer.
            h0 (Tensor | None): initial hidden state, Tensor of shape (batch, hidden_size) or left None.
                If left None, the net will automatically use the previous final hidden state stored in the cache if selected.
            c0 (Tensor | None): initial cell state, Tensor of shape (batch, hidden_size) or left None.
                If left None, the net will automatically use the previous final hidden state stored in the cache if selected.
            
        Returns:
            Tensor: Output tensor of shape (batch_size, hid_features) after applying the RNN transformation.
            or 
            A tuple of Tensors of 3 having outputs, the hidden state, and the cell state.

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
            if self._cache.get('h0_final', None) is not None and self.accumulate_hidden == True:
                h0 = self._cache['h0_final']
                
        # If c0 is given, it must have (batch_size, hid_features) size
        if c0 is not None:
            if isinstance(c0, Tensor) == False:
                raise ValueError(f"In performing forward(), input `c0` must be in a MML `Tensor` format but you have {type(c0)}")
            if c0.shape != (x.shape[0], self.hid_features):
                raise ValueError(f"In performing forward(), if given an input `c0`, it must equal to (batch_size, hidden_size) but you have {c0.shape}")
        # Else if it has been stored previously, reuse it
        else:
            if self._cache.get('c0_final', None) is not None and self.accumulate_hidden == True:
                h0 = self._cache['c0_final']
        
        # Save input for backward
        self.__setattr__("input", x)
        
        # Rrcord the critical shapes
        batch_size, seq_len, *_ = x.shape
        
        # Initialize hidden state
        if h0 is None:
            h = Tensor.zeros((batch_size, self.hid_features), backend=self.backend, dtype=self.dtype, device=self.device)
        else:
            h = h0
            
        # Initialize cell state
        if c0 is None:
            c = Tensor.zeros((batch_size, self.hid_features), backend=self.backend, dtype=self.dtype, device=self.device)
        else:
            c = c0
            
        # Cache lists for backward (to compute gradients)
        window = self.tbptt_steps if (self.tbptt_steps is not None and self.tbptt_steps > 0) else seq_len
        x_cache = deque(maxlen=window)
        i_cache = deque(maxlen=window)
        f_cache = deque(maxlen=window)
        g_cache = deque(maxlen=window)
        o_cache = deque(maxlen=window)
        i_lin_cache = deque(maxlen=window)
        f_lin_cache = deque(maxlen=window)
        g_lin_cache = deque(maxlen=window)
        o_lin_cache = deque(maxlen=window)
        h_cache = deque(maxlen=window)
        c_cache = deque(maxlen=window)
        
        # Cache the initial state
        if self.training == True:
            self._cache['h0'] = deepcopy(h)
            self._cache['c0'] = deepcopy(c)
            
        # Prepare output list if return_sequences
        outputs = [] if self.return_sequences == True else None
        
        # Forward propagate through time
        for t in range(seq_len):
            
            # Crcord input at time t, shape (batch, input_size)
            xt = x[:, t, ...]  
            x_cache.append(xt)

            # Linear combinations for all gates: (batch, 4*H)
            # z_t = [i_lin, f_lin, g_lin, o_lin] = X_t * W_ih^T + h_prev * W_hh^T + (bias_ih + bias_hh)
            z_t = (xt @ self._parameters["weight_ih"].data.transpose()) + (h @ self._parameters["weight_hh"].data.transpose()) + (self._parameters["bias_ih"].data + self._parameters["bias_hh"].data)
            
            # Split combined linear outputs into each gate pre-activation (each of shape (batch, H))
            i_lin, f_lin, g_lin, o_lin = torch.split(z_t, self.hid_features, dim=1)
            
            # Apply activations to get gate values
            i_t = i_lin.sigmoid()           # forget gate
            f_t = f_lin.sigmoid()           # forget gate
            g_t = g_lin.tanh()              # candidate gate (cell input)
            o_t = o_lin.sigmoid()           # output gate
                        
            # Compute new cell state and hidden state
            c_t = f_t * c + i_t * g_t  # (batch, hidden), formula: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
            h_t = o_t *c_t.tanh()      # (batch, hideen), formula: h_t = o_t ⊙ tanh(c_t)

            # Save gate activations and states in cache for backprop
            i_cache.append(i_t)
            f_cache.append(f_t)
            g_cache.append(g_t)
            o_cache.append(o_t)
            i_lin_cache.append(i_lin)
            f_lin_cache.append(f_lin)
            g_lin_cache.append(g_lin)
            o_lin_cache.append(o_lin)
            c_cache.append(c_t)
            h_cache.append(h_t)

            # Append to outputs if needed
            if self.return_sequences == True:
                outputs.append(deepcopy(h_t))

            # Prepare next step
            h, c = h_t, c_t

            # TBPTT: Truncated Back Propagation Through Time
            # if we've reached tbptt_steps, detach hidden state
            if self.tbptt_steps is not None and self.tbptt_steps > 0:
                
                # Detach at the boundary of each segment
                if (t + 1) % self.tbptt_steps == 0:
                    # Detach the hidden state to prevent backprop beyond this point
                    # It is only used for autograd purposes
                    h = h.detach()  
                    c = c.detach()
                    
        # Save cache for backward
        if self.training == True:
            self._cache['x'] = x_cache
            self._cache['i'] = i_cache
            self._cache['f'] = f_cache
            self._cache['g'] = g_cache
            self._cache['o'] = o_cache
            self._cache['i_lin'] = i_lin_cache
            self._cache['f_lin'] = f_lin_cache
            self._cache['g_lin'] = g_lin_cache
            self._cache['o_lin'] = o_lin_cache
            self._cache['c'] = c_cache
            self._cache['h'] = h_cache
            self._cache['seq_len'] = seq_len
            self._cache['batch_size'] = batch_size
            self._cache['h0_final'] = h  
            self._cache['c0_final'] = c 
        # h: final hidden state after last step (stored for continuation reuse)
        # c: final cell state after last step  (stored for continuation reuse)

        # Prepare output data
        if self.return_sequences == True:
            # Stack outputs list to tensor: (batch, seq_len, hidden)
            outputs_tensor = outputs[0].stack(*outputs[1:], axis=1)
        else:
            # Last output is the last hidden state
            outputs_tensor = h

        if self.return_state == True:
            # Return output and final hidden state
            return outputs_tensor, h, c
        else:
            # Return just the output tensor
            return outputs_tensor
    
    def backward(self, grad_output: Tensor, grad_h: Tensor | None = None, grad_c: Tensor | None = None) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
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
                     - This is usually None unless the LSTM's final state is fed into another layer.
            grad_c (Tensor | None): Gradient of loss w.r.t. the final cell state.
                     - This is usually None unless the LSTM's final state is fed into another layer.

        Returns:
            Tuple of 
                (Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers, shape (batch, seq_len, input_size) tensor.
                 Tensor: Gradient with respect to the final hidden state, ∂L/∂h0.
                 Tensor: Gradient with respect to the final cell state, ∂L/∂c0.
                 )
        Raises:
            ValueError: If `grad_output` is not a valid MML.Tensor object.
            ValueError: If `grad_h` is given but not a valid MML.Tensor object.
            ValueError: If `grad_c` is given but not a valid MML.Tensor object.
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
        
        # Type check, if given, grad_c must be an instance of Tensor
        if grad_c is not None:
            if isinstance(grad_c, Tensor) == False:
                raise ValueError(f"In performing backward(), if given `grad_c`, it must be in a MML `Tensor` format but you have {type(grad_c)}")
        
        # Retrieve some dimension values
        x_cache = self._cache.get('x')
        seq_len = self._cache.get('seq_len')
        batch_size = self._cache.get('batch_size') 
        
        # Initialize gradients for inputs and initial states
        grad_x_all = Tensor.zeros((batch_size, seq_len, self.in_features), backend=self.backend, dtype=self.dtype, device=self.device)
        grad_h_prev = Tensor.zeros((batch_size, self.hid_features), backend=self.backend, dtype=self.dtype, device=self.device)  # dL/dh_{t-1}
        grad_c_prev = Tensor.zeros((batch_size, self.hid_features), backend=self.backend, dtype=self.dtype, device=self.device)  # dL/dc_{t-1}
  
        # If return_sequences=False, expand grad_output to sequence length (
        # only last timestep has non-zero grad)
        if self.return_sequences == False:
            # grad_output is (batch, hidden_size) for last output
            # We create an array of shape (batch, seq_len, hidden_size) with zeros except last timestep
            go = Tensor.zeros((batch_size, seq_len, self.hid_features), backend=self.backend, dtype=self.dtype, device=self.device)
            go[:, -1, ...] = grad_output  # only last time step has incoming grad
            grad_output = go
        
        # If an external grad_h is provided, add it
        if grad_h is not None:
            grad_output[:, -1, ...] += grad_h
        
        # If an external grad_c is provided, add it
        if grad_h is not None:
            grad_c_prev += grad_h
        
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
            
            # Current step's hidden and cell from cache
            h_t = self._cache['h'][t]    # shape: (batch, hidden)
            c_t = self._cache['c'][t]    # shape: (batch, hidden)
            
            # Previous cell state and hidden state
            if t == 0:
                c_prev_t = self._cache['c0']
                h_prev_t = self._cache['h0']
            else:
                c_prev_t = self._cache['c'][t-1]
                h_prev_t = self._cache['h'][t-1]
                
            # Gradients from output of this timestep (dL/dh_t)
            d_h = grad_output[:, t, ...] + grad_h_prev
            
            # Gradient from current cell output portion (dL/dc_t from h_t path)
            # h_t = o_t * tanh(c_t) -> dL/dc_t (partial) = dL/dh_t * o_t * (1 - tanh(c_t)^2)
            o_t = self._cache['o'][t]
            tanh_c_t = c_t.tanh()
            d_c = grad_c_prev + (d_h * o_t * (1 - tanh_c_t * tanh_c_t))
            
            # dL/do_t = dL/dh_t * tanh(c_t)
            d_o = d_h * tanh_c_t
            
            # Now split d_c into contributions
            i_t = self._cache['i'][t]
            f_t = self._cache['f'][t]
            g_t = self._cache['g'][t]
            # dL/di_t = d_c * g_t
            d_i = d_c * g_t
            # dL/df_t = d_c * c_{t-1}
            d_f = d_c * c_prev_t
            # dL/dg_t = d_c * i_t
            d_g = d_c * i_t
            # dL/dc_{t-1} for next iteration (grad_c_prev for next step)
            # c_t = f_t * c_{t-1} + ... -> partial derivative w.rt c_{t-1} is f_t
            grad_c_prev = d_c * f_t  # (this becomes d_c for previous time step)
            
            # Backprop through gate activations
            # i_t = sigmoid(i_lin), etc. Compute gradients w.rt pre-activation: d*_lin
            # sigmoid' = s*(1-s); tanh' = 1 - g^2
            i_lin = self._cache['i_lin'][t]
            f_lin = self._cache['f_lin'][t]
            o_lin = self._cache['o_lin'][t]
            g_lin = self._cache['g_lin'][t]
            d_i_lin = d_i * i_t * (1 - i_t)
            d_f_lin = d_f * f_t * (1 - f_t)
            d_o_lin = d_o * o_t * (1 - o_t)
            d_g_lin = d_g * (1 - g_t * g_t)

            # Concatenate gate gradients for linear part: (batch, 4H)
            d_z = d_i_lin.hstack(d_f_lin, d_g_lin, d_o_lin)
            
            # Gradients w.rt weight matrices and biases
            # weight_ih and weight_hh were used as: i_lin = X_t * W_ih^T + ..., so:
            # grad_weight_ih += d_z^T * X_t
            # grad_weight_hh += d_z^T * h_prev_t
            self._parameters["weight_ih"].grad += d_z.transpose() @ self._cache['x'][t]
            self._parameters["weight_hh"].grad += d_z.transpose() @ h_prev_t
            
            # Accumulate bias gradients (each just sum of d_z, since bias is added to each output)
            if self.has_bias == True:
                self._parameters["bias_ih"].grad += d_z.sum(axis=0)
                self._parameters["bias_hh"].grad += d_z.sum(axis=0)

            # Gradients w.rt input x_t and hidden state h_{t-1}
            # dX_t = d_z * W_ih (because z = X * W_ih^T -> dX = dZ * W_ih)
            grad_x_all[:, t, ...] = d_z @ self._parameters["weight_ih"].data
            # dH_prev = d_z * W_hh
            grad_h_prev = d_z @ self._parameters["weight_hh"].data
            
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
        
        return grad_x_all, grad_h_prev, grad_c_prev
    
    def __repr__(self):
        return f"nn_Layer_LSTMCell(shape: ({self.in_features}, {self.hid_features}) with{'out' if self.has_bias == False else ''} bias)."
      
        
# Implementation of a Stacked LSTM Layer using TBPTT  
class nn_Layer_StackedLSTM(nn_Module):
    
    """
    Stacked LSTM Implementation
    
    This class implements a Long Short-Term Memory (LSTM) cell, designed to process
    sequential data by maintaining and updating both hidden and cell states at each time step. 
    It contains multiple weight matrices (input, forget, cell, output gates) and corresponding 
    bias vectors stored in nn_Parameter containers, enabling the computation of forward 
    passes through sequences while supporting gradient-based learning via backward passes. 
    The LSTM architecture introduces gate mechanisms to control information flow, allowing
    for better capture of long-term dependencies compared to standard RNNs.
        
    Note, TBPTT should be implemented externally with segmentation and carrying (manually reset_hidden).
    """
    
    __attr__ = "MML.nn_Layer_StackedLSTM"   

    def __init__(self, 
                 in_features: int = 1, 
                 hid_features: int = 1, 
                 num_layers: int = 1,
                 has_bias: str = True,
                 init_scale: float = 0.1,
                 tbptt_steps: int | None = None,
                 return_sequences: bool = False, 
                 return_state: bool = False,
                 accumulate_hidden: bool = False,
                 gradient_clipping: float = 5.0,
                 *,
                 module_name: str = "nn_Layer_LSTMCell", 
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        An LSTM (Long Short Term Machine) cell processes sequences by maintaining
            a hidden state that is updated at each time step.
            
        Structure: 
            - **Input gate**: Controls how much new information is added to the cell state.  
              $$ i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) $$  
            - **Forget gate**: Determines what information to discard from the cell state.  
              $$ f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) $$  
            - **Cell gate (candidate)**: Computes a candidate value for the cell state.  
              $$ \tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c) $$  
            - **Output gate**: Regulates what part of the cell state is output as the hidden state.  
              $$ o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) $$  
        
        Parameters:
            in_features: int, The number of input features for this layer. Defaults to 1.
            hid_features: int, The number of hidden features for this layer. Defaults to 1.
            num_layers: int, The number of LSTM cells stacked together (h(n-1) becomes x in stack n).
            has_bias: str, A flag indicating whether to include a bias term. Valid values are "True" or "False".
                    If set to "True", the layer includes an additive bias term (b). Defaults to "True".
            tbptt_steps: int | None, the number of steps performing Truncated BPTT (TBPTT), 
                    if None, then use true BPTT. Defaults: None.
                    Note, TBPTT is not fully implemented. Please use BPTT instead.
            return_sequences: bool, should the layers return only the final hidden state, or the full sequence 
                    of hidden states as well. Defaults to False.
            return_state: bool, should the layers return output and final hidden state. Defaults to False.
            accumulate_hidden: bool, should the network accumulate hidden state (when processing segments) or clear it before next forward.
            gradient_clipping: float, The factor of performing gradient clipping to avoid gradient expolosion. Defaults to 5.0.
            module_name: str, The name of the module instance. Defaults to "nn_Layer_StackedLSTM".
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
        
        # Record shapes etc
        self.__setattr__("in_features", in_features)
        self.__setattr__("hid_features", hid_features)
        self.__setattr__("num_layers", num_layers)
        self.__setattr__("has_bias", has_bias)
        self.__setattr__("init_scale", init_scale)
        self.__setattr__("tbptt_steps", tbptt_steps)
        self.__setattr__("return_sequences", return_sequences)
        self.__setattr__("return_state", return_state)
        self.__setattr__("accumulate_hidden", accumulate_hidden)
        self.__setattr__("gradient_clipping", gradient_clipping)
        
        # Initialize stacked RNN components, each returning full sequence + its final state
        for i in range(num_layers):
            self.__setattr__("LSTM_Layer_"+str(i), 
                             nn_Layer_LSTMCell(
                                 in_features=in_features if i == 0 else hid_features,
                                 hid_features=hid_features,
                                 has_bias=has_bias,
                                 init_scale=init_scale,
                                 tbptt_steps=tbptt_steps,
                                 return_sequences=True,
                                 return_state=True,
                                 accumulate_hidden=accumulate_hidden,
                                 gradient_clipping=gradient_clipping,
                                 module_name=module_name+"_"+"LSTM_Layer_"+str(i),
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

    def forward(self, x: Tensor, h0: List[Tensor] | None = None, c0: List[Tensor] | None = None) -> Tensor| Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Compute the layer output: h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh}) \tag{1} and return it.
        You can safely choose NOT to return hidden states since we have memorized them if you do not pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)) to be transformed by the layer.
            h0 (List[Tensor] | None): initial hidden state, List of Tensors of shape (batch, hidden_size) or left None.
            c0 (List[Tensor] | None): initial cell state, List of Tensors of shape (batch, hidden_size) or left None.
            
        Returns:
            Tensor: Output tensor of shape (batch_size, hid_features) after applying the RNN transformation.
            or 
            A tuple of 3 having outputs and the List of Tensors containing hidden state and cell state.

        Raises:
            ValueError: If the input `x` is not a valid MML.Tensor object.
            ValueError: If the input `h0` is given but not a list of Tensors.
            ValueError: If the input `c0` is given but not a list of Tensors.
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
            
        # If c0 is given, it must have (batch_size, hid_features) size
        if c0 is not None:
            if isinstance(c0, list) == False:
                raise ValueError(f"In performing forward(), input `c0` must be in a List of MML `Tensor` format but you have {type(c0)}")
            if isinstance(c0[0], Tensor) == False:
                raise ValueError(f"In performing forward(), if given an input `c0`, it must be a list of MML `Tensor`s but you have {type(c0[0])}")
             
        # Clear cache to release spaces
        self.clear_cache()
             
        # Prepare output and final states/cells
        output = x
        final_states = []
        final_cells = []

        # Perform RNN Layer by layer
        for idx, key in enumerate(self._modules.keys()):
            h = h0[idx] if (h0 is not None) else None
            c = c0[idx] if (c0 is not None) else None
            # Perform a forward pass for each layer sequentially
            output, h_n, c_n = self._modules[key].forward(output, h, c)
            final_states.append(h_n)
            final_cells.append(c_n)
            
        # if only last step is wanted, slice it off the sequence
        if self.return_sequences == False:
            # (batch, hidden_size)
            output = output[:, -1, ...] 
            
        # If return state, then return a list of states
        if self.return_state == True:
            return output, final_states, final_cells
        # Otherwise, only the output
        else:
            return output
        
    def backward(self, grad_output: Tensor, grad_h: List[Tensor] | None = None, grad_c: List[Tensor] | None = None) -> Tensor | Tuple[Tensor, List[Tensor], List[Tensor]]:
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
                     - This is usually None unless the LSTM's final state is fed into another layer.
                     - If used, it is List of Tensors with shape (batch, hidden_size)
            grad_c (List[Tensor] | None): Gradient of loss w.r.t. the final cell state.
                     - This is usually None unless the LSTM's final state is fed into another layer.
                     - If used, it is List of Tensors with shape (batch, hidden_size)
                     
        Returns:
            Tuple of 
                (Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers, shape (batch, seq_len, input_size) tensor.
                 List[Tensor]: Gradients with respect to each final hidden state, List[∂L/∂h0] (batch, hidden).
                 List[Tensor]: Gradients with respect to each final cell state, List[∂L/∂c0] (batch, hidden).
                 )
        Raises:
            ValueError: If `grad_output` is not a valid MML.Tensor object.
            ValueError: If `grad_h` is given but not a valid list of MML.Tensor object.
            ValueError: If `grad_c` is given but not a valid list of MML.Tensor object.
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
        
        # Type check, if given, grad_c must be an instance of Tensor
        if grad_c is not None:
            if isinstance(grad_c, list) == False:
                raise ValueError(f"In performing backward(), if given `grad_c`, it must be in a List of MML `Tensor`s format but you have {type(grad_c)} with each {type(grad_c[0])}")
            if isinstance(grad_c[0], Tensor) == False:
                raise ValueError(f"In performing backward(), if given `grad_c`, it must be in a List of MML `Tensor`s format but you have {type(grad_c)} with each {type(grad_c[0])}")
        
        # prepare per-layer state gradients
        if grad_h is None:
            grad_h = [None] * self.num_layers

        # prepare per-layer cell gradients
        if grad_c is None:
            grad_c = [None] * self.num_layers
                 
        # grad flowing into top layer's output
        next_grad_out = grad_output
        grad_h0_list = [None] * self.num_layers
        grad_c0_list = [None] * self.num_layers

        # walk backward through layers
        keys = list(self._modules.keys())
        for i in reversed(range(self.num_layers)):
            layer    = self._modules[keys[i]]
            grad_h_n = grad_h[i]
            grad_c_n = grad_c[i]
            grad_out  = next_grad_out

            # if only last-step output was used externally, expand it
            if self.return_sequences == False and i == self.num_layers - 1:
                seq_len = layer._cache["seq_len"]
                batch_size = layer._cache["batch_size"]
                hidden = layer.hid_features
                full = Tensor.zeros((batch_size, seq_len, hidden), backend=self.backend, dtype=self.dtype, device=self.device)
                full[:, -1, ...] = grad_out
                grad_out = full

            # call sub-layer backward (must return three)
            grad_x, grad_h0, grad_c0 = layer.backward(grad_out, grad_h_n, grad_c_n)

            grad_h0_list[i] = grad_h0
            grad_c0_list[i] = grad_c0
            next_grad_out = grad_x

        # next_grad_out is gradient w.rt the very input x
        # grad_h0_list is gradient w.rt the hidden state of each layer
        # grad_c0_list is gradient w.rt the cell state of each layer
        return next_grad_out, grad_h0_list, grad_c0_list

    def __repr__(self):
        return f"nn_Layer_StackedLSTM(shape: ({self.in_features}, {self.hid_features}), num_layers: {self.num_layers} with{'out' if self.has_bias == False else ''} bias)."
    

# Alias for nn_Layer_StackedLSTM
LSTM = nn_Layer_StackedLSTM
StackedLSTM = nn_Layer_StackedLSTM


# Test case for Stacked RNN
if __name__ == "__main__":
    
    import torch
    from nn import Tensor
    from nn import Dense
    from nn import StackedLSTM
    from nn import Sigmoid, ReLU
    from nn import Softmax
    from nn import Module, nn_Module
    from nn import RMSE, MSE, BinaryCrossEntropy
    from nn import Adam
    from nn import Evaluator
    
    backend = "torch"
    device = "cuda"
    
    # Load dataset
    import pandas as pd
    df = pd.read_csv("./dataset/spx-choice-daily-2005-simple.csv")
    df = df.drop(["Date"], axis=1)
    X = df.drop(["Next Day"], axis=1)
    y = df[["Next Day"]]
    X[["Open_LMA", "High_LMA", "Low_LMA", "Close_LMA"]] = X[["Open_LMA", "High_LMA", "Low_LMA", "Close_LMA"]] / 100
    X = Tensor(X.to_numpy(), backend=backend, device=device, dtype=torch.float32)
    y = Tensor(y.to_numpy(), backend=backend, device=device, dtype=torch.float32)
    
    # Create windows
    Xw, yw = Module.make_rolling_window(X, y, window_size=10)
    Xw = Xw[8000:]
    yw = yw[8000:]
    
    class test_lstm(nn_Module):
        def __init__(self,
                     input_size, hidden_size, 
                     num_layers=1, tbptt_steps=None, **kwargs):
            
            super().__init__(**kwargs)
            
            # register the stacked lstm as a submodule
            self.stacked = StackedLSTM(
                input_size, hidden_size,
                num_layers=num_layers,
                tbptt_steps=tbptt_steps,
                return_sequences=False,
                return_state=False,  # State is memorized
                **kwargs
            )
            # register the final dense head
            self.head = Dense(hidden_size, 1, **kwargs)
            self.actv = Sigmoid(**kwargs)
    
        def forward(self, x):
            out = self.stacked(x)
            out = self.head(out)
            out = self.actv(out)
            return out
        
    x = test_lstm(input_size = 12, hidden_size = 16, backend = backend, device = device)
    
    x.train()
    
    crit = BinaryCrossEntropy(backend=backend, device=device)
    optm = Adam(x.parameters(), lr=2e-4)
    
    for i in range(1000):
        out = x.forward(Xw)
        
        # Calculate loss
        loss = crit(out, yw);
        print(loss.to_list())
        lossgrad = crit.backward()
    
        # Test backward
        x.backward(lossgrad)
        
        # Apply sgd
        optm.step()
        
        # Apply zero grad
        x.zero_grad()
        
        
    x.eval()
    x.forward(Xw)