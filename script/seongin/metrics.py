import numpy as np

def smape_metric(y_true, y_pred):
    """
    DACON 대회용 SMAPE 평가 지표
    - 실제 매출 수량이 0인 경우는 평가에서 제외합니다.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = y_true != 0
    
    if not np.any(mask):
        return 0.0

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    
    return np.mean(ratio) * 100
