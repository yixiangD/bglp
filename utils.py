import keras.backend as K

def RMSE(output, target):
    return K.sqrt(K.mean((output - target) ** 2))

def MAE(true, preds):
    true = true.flatten()
    preds = preds.flatten()
    return sum(abs(true-preds))/true.shape[0]
