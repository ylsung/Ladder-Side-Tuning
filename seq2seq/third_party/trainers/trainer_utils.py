import numpy as np 
from typing import Union, NamedTuple, Tuple, Dict, Any   

class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
        data_info: (:obj:`Dict[str, Any]`): Extra dataset information, one requires
        to performs the evaluation. The data_info is a dictionary with keys from
        train, eval, test to specify the data_info for each split of the dataset.
    """
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray
    data_info: Dict[str, Any]

