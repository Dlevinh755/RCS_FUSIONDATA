# Export core components from LightGNN
from .LightGNN.LightGNN import train as LightGNN_train
from .CAMRec.train import train as CAMRec_train


__all__ = [
    "LightGNN_train",
    "CAMRec_train",
]
