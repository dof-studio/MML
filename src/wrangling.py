# wrangling.py
#
# Utilities for Feature Engineering
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None
    
from copy import deepcopy
from typing import Any, List, Dict, Tuple

from objtyp import Object
from tensor import Tensor
from matrix import Matrix

from baseml import MLBase


# Base Class for Data Wrangling
class BaseDataWrangling(MLBase):
    
    __attr__ = "MML.BaseDataWrangling"
    
    def __init__(self, *, _method: str | None = None, _parameters: Dict = {}, _feature_names: Dict | None = None, **kwargs):
        """
        Initializer (constructor) for the base DataWrangling.
        
        Attributes:
            --------
            self._method = _method: str, the internal type of wrangling used.
            self._parameters = _parameters: dict, the parameters to save the args in any wrangling.
            self._feature_names = feature_names: dict, the feature names with respect to axis (like column) index.
            
        Raises:
            --------
            ValueError, if _method is not an instance of str.
        """
        # Check and save the method
        if isinstance(_method, str):
            self._method = _method
        else:
            raise ValueError("The internal method `_method` parameter must be an instance of str.")
        
        # Save the internal parameters
        self._parameters = deepcopy(_parameters)
        
        # Save the feature names
        self._feature_names = deepcopy(_feature_names)
            
    def fit(self, X: Matrix | Tensor, **kwargs):
        """
        Fit the data wrangler by some data. Note, just fit the model but without any transform happened.
        
        Parameters:
            --------
            X: Matrix | Tensor, the original data to be fitted.
            
        Returns:
            --------
            self
        """
        raise NotImplementedError("Fit is NOT implemented in the base class.")
              
    def transform(self, X: Matrix | Tensor, **kwargs) -> Tuple[Matrix | Tensor, Dict]:
        """
        Transform the data when the wrangling model is fitted.
        
        Parameters:
            --------
            X: Matrix | Tensor, the original data to be tansformed.
            
        Returns:
            --------
            Tuple[Matrix | Tensor, Dict]: the tuple of (Data Transformed, new dict of names if given else None)
        """
        raise NotImplementedError("Transform is NOT implemented in the base class.")    
        
    def fit_transform(self, X: Matrix | Tensor, **kwargs) -> Tuple[Matrix | Tensor, Dict]:
        """
        Fit the data wrangler and transform the data when the wrangling model is fitted.
        
        Parameters:
            --------
            X: Matrix | Tensor, the original data to be fitted and transformed.
            
        Returns:
            --------
            Tuple[Matrix | Tensor, Dict]: the tuple of (Data Transformed, new dict of names if given else None)
        """
        raise NotImplementedError("Transform is NOT implemented in the base class.")    
        
    def inverse_transform(self, X: Matrix | Tensor, **kwargs) -> Tuple[Matrix | Tensor, Dict]:
        """
        Do the inverse transform to get the original untransformed data.
        * Note, this is not necessarily required to be implemented by children classes.
        
        Parameters:
            --------
            X: Matrix | Tensor, the original data to be inversely transformed.
            
        Returns:
            --------
            Tuple[Matrix | Tensor, Dict]: the tuple of (Data Inversely Transformed, new dict of names if given else None)
        """
        raise NotImplementedError("Transform is NOT implemented in the base class.")    
        
    def __repr__(self):
        return "BaseDataWrangling(Abstract Class)."
    
    
# Tabular Interactor (Making Tabular Data Interactive with each other/ some others)
class TabularInteractor(BaseDataWrangling):
    
    __attr__ = "MML.TabularInteractor"
    
    # pass
    # Not implemented

