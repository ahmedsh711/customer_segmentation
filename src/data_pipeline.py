import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer

def get_preprocessing_pipeline():
    log_transformer = FunctionTransformer(np.log1p, validate=False)
    scaler = RobustScaler()
    
    pipeline = Pipeline(steps=[
        ('log', log_transformer),
        ('scale', scaler)
    ])
    
    return pipeline