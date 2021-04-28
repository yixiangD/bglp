import numpy as np
import keras.backend as K

def RMSE(output, target):
    return K.sqrt(K.mean((output - target) ** 2))

def MAE(true, preds):
    true = np.reshape(true, [-1])
    preds = np.reshape(preds, [-1])
    return np.sum(np.abs(true-preds))/true.shape[0]
