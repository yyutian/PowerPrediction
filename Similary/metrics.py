# import numpy as np
import numpy as np

def compute_error(y_true, y_pre):
    #corr = np.corrcoef(predicted, trues)[0,1]
    #mae = np.mean(np.abs(predicted - trues))
    #rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    mse=np.mean(np.square(y_pre-y_true))
    rmse = np.sqrt(np.mean((y_pre-y_true)**2))
    #rrse = np.sqrt(np.sum((predicted - trues)**2) / np.sum((trues - np.mean(trues))**2))
    #mape = np.mean(np.abs((predicted - trues) / trues)) * 100
    r2 = max(0, abs(1 - np.sum((y_pre - y_true)**2) / np.sum((y_true - np.mean(y_true))**2)))
    return  mse,rmse,r2
def rmse(y_true, y_pre):
    return np.sqrt(np.mean((y_pre-y_true)**2))
def mse(y_true, y_pre):
    return np.mean(np.square(y_pre - y_true))

