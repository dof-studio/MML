# __init__.py
#
# Import All Modules
# From MML Library by Nathmath

# Macro------------------------------------------------------------------------
__author__ = "DOF Studio/Nathmath"

__version__ = "0.0.4.0"

__all__ = [
    # aggregation.py
    'Aggregation', 'RegressionAggregation', 'ClassificationAggregation', 
    # baseml.py
    'MLBase', 'Regression', 'Classification', 
    # categorical_encoder.py
    'CategoricalEncoder', 
    # dump.py
    "save", "load",
    # ensemble.py
    'Ensemble', 'Bagging', 'Boosting', 
    # gbm.py
    'BaseGradientBoosting', 'GradientBoostingModel', 'GBM',
    # linear.py
    'BaseLinearModels', 'BaseSingleValueLinearRegression', 'BaseMultiValueLinearRegression', 'ClosedFormSingleValueRegression', 'GradientDescendSingleValueRegression', 
    # lm.py
    'OrdinaryLinearRegression', 'LM', 'LR', 'OLS', 'GLS',
    # matrix.py
    'Matrix', 
    # metrics.py
    'BaseMetrics', 'RegressionMetrics', 'ClassificationMetrics', 'BinaryClassificationMetrics', 'MultiClassificationMetrics', 
    
    # nn.py
    # ----------------------------------------------------------------
    # nn_activation.py
    'nn_Activation_ReLU', 'nn_Activation_LeakyReLU', 'nn_Activation_Sigmoid', 'nn_Activation_Tanh', 'nn_Activation_Softmax', 
    'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'Softmax',
    # nn_base.py
    'nn_Base', 
    # nn_layer_dense.py
    'nn_Layer_Dense', "Dense",
    # nn_layer_dropout.py
    'nn_Layer_Dropout', 'Dropout',
    # nn_layer_flatten.py 
    'nn_Layer_Flatten', 'Flatten',
    # nn_layer_lstm.py    
    'nn_Layer_LSTMCell', 'nn_Layer_StackedLSTM', 'LSTM', 'StackedLSTM', 
    # nn_layer_rnn.py
    'nn_Layer_RNNCell', 'nn_Layer_StackedRNN', 'RNN', 'StackedRNN',
    # nn_loss.py
    'nn_Loss_BaseLoss', 'nn_Loss_MSE', 'nn_Loss_RMSE', 'nn_Loss_MAE', 'nn_Loss_BinaryCrossEntropy', 'nn_Loss_MultiCrossEntropy', 
    'MSE', 'RMSE', 'MAE', 'BinaryCrossEntropy', 'MultiCrossEntropy',
    # nn_module.py
    'nn_BaseModule', 'nn_Module', 'Module',
    # nn_optimizer.py
    'nn_Optm_BaseOptimizer', 'nn_Optm_SGD', 'nn_Optm_Adam', 'nn_Optm_AdamW',
    'SGD', 'Adam', 'AdamW',
    # nn_parameter.py
    'nn_Parameter', 
    # nn_sinterf_evaluator.py
    'nn_SInterf_Evaluator', 'Evaluator',
    # ----------------------------------------------------------------

    # objtyp.py
    'Object', 
    # optimizer.py
    'GradientOptimizer', 'optimize',
    # pca.py
    'PCA', 
    # random_forest.py
    'RandomForest', 'RF',
    # scaling.py
    'Scaling', 
    # svm.py
    'SVM', 'SVC',
    # tensor.py
    'Tensor', 
    # threadpool.py
    'ThreadPool', 'Mutex', 
    # time_series.py
    'BaseTimeSeriesModel', 'ARIMAX', 
    # tree.py
    'BaseTree', 'CART',
    # tree_wrapper.py
    'LRTW', 
    # wrangling.py
    'BaseDataWrangling', 'TabularInteractor', 
]

# Production ------------------------------------------------------------------
mml_production = True  # If production, you must import MML from your python's site packages
                       # Else, ensure MML's source code must be in your working directory

# Import-----------------------------------------------------------------------

# aggregation.py
if mml_production:
    import mml.aggregation
    from mml.aggregation import Aggregation, RegressionAggregation, ClassificationAggregation
else:
    import aggregation
    from aggregation import Aggregation, RegressionAggregation, ClassificationAggregation

# baseml.py
if mml_production:
    import mml.baseml
    from mml.baseml import MLBase, Regression, Classification
else:
    import baseml
    from baseml import MLBase, Regression, Classification

# categorical_encoder.py
if mml_production:
    import mml.categorical_encoder
    from mml.categorical_encoder import CategoricalEncoder
else:
    import categorical_encoder
    from categorical_encoder import CategoricalEncoder

# dump.py
if mml_production:
    import mml.dump
    from mml.dump import save, load
else:
    import dump
    from dump import save, load

# ensemble.py
if mml_production:
    import mml.ensemble
    from mml.ensemble import Ensemble, Bagging, Boosting
else:
    import ensemble
    from ensemble import Ensemble, Bagging, Boosting

# gbm.py
if mml_production:
    import mml.gbm
    from mml.gbm import BaseGradientBoosting, GradientBoostingModel, GBM
else:
    import gbm
    from gbm import BaseGradientBoosting, GradientBoostingModel, GBM

# linear.py
if mml_production:
    import mml.linear
    from mml.linear import BaseLinearModels, BaseSingleValueLinearRegression, BaseMultiValueLinearRegression
    from mml.linear import ClosedFormSingleValueRegression, GradientDescendSingleValueRegression
else:
    import linear
    from linear import BaseLinearModels, BaseSingleValueLinearRegression, BaseMultiValueLinearRegression
    from linear import ClosedFormSingleValueRegression, GradientDescendSingleValueRegression

# lm.py
if mml_production:
    import mml.lm
    from mml.lm import OrdinaryLinearRegression, LM, LR, OLS, GLS
else:
    import lm
    from lm import OrdinaryLinearRegression, LM, LR, OLS, GLS

# matrix.py
if mml_production:
    import mml.matrix
    from mml.matrix import Matrix
else:
    import matrix
    from matrix import Matrix

# metrics.py
if mml_production:
    import mml.metrics
    from mml.metrics import BaseMetrics, RegressionMetrics
    from mml.metrics import ClassificationMetrics, BinaryClassificationMetrics, MultiClassificationMetrics
else:
    import metrics
    from metrics import BaseMetrics, RegressionMetrics
    from metrics import ClassificationMetrics, BinaryClassificationMetrics, MultiClassificationMetrics

# nn.py 
if mml_production:
    import mml.nn
    # Modules
    from mml.nn import Module, nn_Module
    # Layers
    from mml.nn import Dense, nn_Layer_Dense
    from mml.nn import Dropout, nn_Layer_Dropout
    from mml.nn import Flatten, nn_Layer_Flatten
    from mml.nn import RNN, StackedRNN, nn_Layer_StackedRNN
    from mml.nn import LSTM, StackedLSTM, nn_Layer_StackedLSTM
    # # Activations
    from mml.nn import ReLU, nn_Activation_ReLU
    from mml.nn import LeakyReLU, nn_Activation_LeakyReLU
    from mml.nn import Sigmoid, nn_Activation_Sigmoid
    from mml.nn import Tanh, nn_Activation_Tanh
    from mml.nn import Softmax, nn_Activation_Softmax
    # Losses
    from mml.nn import MSE, nn_Loss_MSE
    from mml.nn import RMSE, nn_Loss_RMSE
    from mml.nn import MAE, nn_Loss_MAE
    from mml.nn import BinaryCrossEntropy, nn_Loss_BinaryCrossEntropy
    from mml.nn import MultiCrossEntropy, nn_Loss_MultiCrossEntropy
    # Optimizers
    from mml.nn import SGD, nn_Optm_SGD
    from mml.nn import Adam, nn_Optm_Adam
    from mml.nn import AdamW, nn_Optm_AdamW
    # SInterf Evaluator
    from mml.nn import Evaluator, nn_SInterf_Evaluator
else:
    import nn
    # Modules
    from nn import Module, nn_Module
    # Layers
    from nn import Dense, nn_Layer_Dense
    from nn import Dropout, nn_Layer_Dropout
    from nn import Flatten, nn_Layer_Flatten
    from nn import RNN, StackedRNN, nn_Layer_StackedRNN
    from nn import LSTM, StackedLSTM, nn_Layer_StackedLSTM
    # # Activations
    from nn import ReLU, nn_Activation_ReLU
    from nn import LeakyReLU, nn_Activation_LeakyReLU
    from nn import Sigmoid, nn_Activation_Sigmoid
    from nn import Tanh, nn_Activation_Tanh
    from nn import Softmax, nn_Activation_Softmax
    # Losses
    from nn import MSE, nn_Loss_MSE
    from nn import RMSE, nn_Loss_RMSE
    from nn import MAE, nn_Loss_MAE
    from nn import BinaryCrossEntropy, nn_Loss_BinaryCrossEntropy
    from nn import MultiCrossEntropy, nn_Loss_MultiCrossEntropy
    # Optimizers
    from nn import SGD, nn_Optm_SGD
    from nn import Adam, nn_Optm_Adam
    from nn import AdamW, nn_Optm_AdamW
    # SInterf Evaluator
    from nn import Evaluator, nn_SInterf_Evaluator

# objtyp.py
if mml_production:
    import mml.objtyp
    from mml.objtyp import Object
else:
    import objtyp
    from objtyp import Object

# optimizer.py
if mml_production:
    import mml.optimizer
    from mml.optimizer import GradientOptimizer, optimize
else:
    import optimizer
    from optimizer import GradientOptimizer, optimize

# pca.py
if mml_production:
    import mml.pca
    from mml.pca import PCA
else:
    import pca
    from pca import PCA

# random_forest.py
if mml_production:
    import mml.random_forest
    from mml.random_forest import RandomForest, RF
else:
    import random_forest
    from random_forest import RandomForest, RF

# scaling.py
if mml_production:
    import mml.scaling
    from mml.scaling import Scaling
else:
    import scaling
    from scaling import Scaling

# svm.py
if mml_production:
    import mml.svm
    from mml.svm import SVM, SVC
else:
    import svm
    from svm import SVM, SVC

# tensor.py
if mml_production:
    import mml.tensor
    from mml.tensor import Tensor
else:
    import tensor
    from tensor import Tensor

# threadpool.py
if mml_production:
    import mml.threadpool
    from mml.threadpool import ThreadPool, Mutex
else:
    import threadpool
    from threadpool import ThreadPool, Mutex

# time_series.py
if mml_production:
    import mml.time_series
    from mml.time_series import BaseTimeSeriesModel, ARIMAX
else:
    import time_series
    from time_series import BaseTimeSeriesModel, ARIMAX

# tree.py
if mml_production:
    import mml.tree
    from mml.tree import BaseTree, CART
else:
    import tree
    from tree import BaseTree, CART

# tree_wrapper.py
if mml_production:
    import mml.tree_wrapper
    from mml.tree_wrapper import LRTW
else:
    import tree_wrapper
    from tree_wrapper import LRTW

# wrangling.py
if mml_production:
    import mml.wrangling
    from mml.wrangling import BaseDataWrangling, TabularInteractor
else:
    import wrangling
    from wrangling import BaseDataWrangling, TabularInteractor
