from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
import numpy as np

def preprocessing_pipeline():
    log_transformer = FunctionTransformer(np.log1p, validate=False)
    scaler = RobustScaler()
    
    pipeline = Pipeline(steps=[
        ('Log Transform', log_transformer),
        ('Scaling', scaler)
    ])
    
    return pipeline