# categorical_encoder.py
#
# A useful encoder to process categorical data as numeric
# From MML Library by Nathmath

import numpy as np
import pandas as pd
from tensor import Tensor
from matrix import Matrix


# Encode categorical data into onehot/multilabel columns
class CategoricalEncoder:
    
    __attr__ = "MML.CategoricalEncoder"
    
    def __init__(self, columns, mode:str='onehot', pad_value=None, sort_func=None):
        """
        Initialize the encoder with parameters settings (without passing in the data).
        
        Parameters:
        - columns: list of column names (categorical columns to be encoded)
        - mode: 'onehot' for one-hot encoding, 'multi' for multi-class integer encoding
        - pad_value: scalar value to fill missing values for all columns or dict with key=column and value=padding value
        - sort_func: function to decide sorting of unique labels. If None, built-in sort() is used.
        """
        if mode not in ['onehot', 'multi']:
            raise ValueError("mode should be either 'onehot' or 'multi'")
        if pad_value is None:
            raise ValueError("pad_value must be something non-None, since we are going to convert them into a numeric value")
        
        self.columns = [columns] if isinstance(columns, str) else columns
        self.mode = mode
        self.pad_value = pad_value
        self.sort_func = sort_func
        # To store mappings from each column to the sorted labels (for multi-class conversion)
        self.label_mappings = {}
        
        # Original and fitted dataframes
        self.original_df = None
        self.fitted_df = None
    
    def _fill_na(self, df: pd.DataFrame, col: str | int):
        """
        Fill missing values for a single column based on pad_value setting.
        """
        if isinstance(self.pad_value, dict):
            fill_value = self.pad_value.get(col, np.nan)
        else:
            fill_value = self.pad_value
        return df[col].fillna(fill_value)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the encoder on the dataframe and transform the given columns.
        It returns a new dataframe with encoded columns.
        
        Returns:
            ---------
            df, pd.DataFrame, the new dataframe contains processed columns.
        """
        new_df = df.copy()
        new_columns = []
        
        for col in self.columns:
            # fill missing values
            new_df[col] = self._fill_na(new_df, col)
            # extract unique values after padding
            unique_labels = new_df[col].unique().tolist()
            # sort using the provided sort function if available
            if self.sort_func is not None:
                ordered_labels = sorted(unique_labels, key=self.sort_func)
            else:
                try:
                    # Attempt to use standard sort for comparable types
                    ordered_labels = sorted(unique_labels)
                except Exception:
                    # Fallback: keep original order if types are not directly comparable
                    ordered_labels = unique_labels
            
            if self.mode == 'onehot':
                # Create new one-hot columns for each unique label
                for label in ordered_labels:
                    new_col_name = f"{col}_{label}"
                    # Generate binary column (1 if equal to label, 0 otherwise)
                    new_df[new_col_name] = (new_df[col] == label).astype(int)
                    new_columns.append(new_col_name)
                # Optionally drop the original column if not needed
                new_df.drop(columns=[col], inplace=True)
            # multi-class integer encoding
            else:  
                # Map each label to an integer (order defines the numeric code starting at 0)
                mapping = {label: i for i, label in enumerate(ordered_labels)}
                self.label_mappings[col] = mapping
                new_df[col] = new_df[col].map(mapping)
                new_columns.append(col)
        
        # If onehot mode, return only newly created columns; if multi mode, return the entire dataframe with transformed columns.
        self.original_df = df.copy()
        self.fitted_df = new_df.copy()
        return new_df

    def fit_transform_self(self, df: pd.DataFrame):
        """
        Fit the encoder on the dataframe and transform the given columns.
        It saves the new dataframe with encoded columns and return the instance of this self.
        
        Returns:
            ---------
            self, CategoricalEncoder, which will be helpful if you intend to further transform this into a Matrix or Tensor.

        """
        ndf = self.fit_transform(df)
        self.fitted_df = ndf
        return self

    def to(self, df:pd.DataFrame | None = None, container:type = Matrix, backend:str = "numpy", *, dtype=None, device=None) -> Matrix | Tensor:
        """
        Convert the saved fitted_df into a Matrix or Tensor object.
        
        Returns:
            ---------
            Matrix | Tensor, converted dataframe.
        """
        if df is None:
            df = self.fitted_df
        return container(df.values, backend=backend, dtype=dtype, device=device)


# Test cases
if __name__ == "__main__":
    
    # Create a sample dataframe
    data = {
        "other": [1,2,3,4,5,6],
        'color': ['red', 'blue', np.nan, 'green', 'red', 'blue'],
        'shape': ['circle', np.nan, 'triangle', 'square', 'circle', 'circle']
    }
    df = pd.DataFrame(data)
    
    # Custom sort function: sort by string length (for demonstration) 
    # (if two strings have equal length, standard lexicographical order is applied)
    def custom_sort(val):
        return len(str(val)), str(val)
    
    # Test one-hot encoding with a scalar pad value.
    encoder_onehot = CategoricalEncoder(columns=['color', 'shape'], mode='onehot', pad_value='missing', sort_func=custom_sort)
    onehot_df = encoder_onehot.fit_transform(df)
    onehot_matrix = encoder_onehot.fit_transform_self(df).to()
    print("One-hot Encoded DataFrame:")
    print(onehot_df)
    
    # Test multi-class encoding with different pad values for each column.
    encoder_multi = CategoricalEncoder(columns=['color', 'shape'], mode='multi', pad_value={'color': 'none', 'shape': 'none'}, sort_func=custom_sort)
    multi_df = encoder_multi.fit_transform(df)
    multi_matrix = encoder_multi.fit_transform_self(df).to()
    print("\nMulti-class Encoded DataFrame:")
    print(multi_df)
    print("\nLabel mappings:")
    print(encoder_multi.label_mappings)
