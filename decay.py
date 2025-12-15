import numpy as np
from typing import Tuple, Union, Dict, Any


class ExponentialDecay:
    def __init__(
        self,
        recency_factor: Union[float, Dict[str, Any]] = 10.0,
        importance_factor: Union[float, Dict[str, Any]] = 0.988,
    ):
        # Handle dictionary input
        if isinstance(recency_factor, dict):
            self.recency_factor = float(recency_factor.get("recency_factor", 10.0))
        else:
            self.recency_factor = float(recency_factor)
            
        if isinstance(importance_factor, dict):
            self.importance_factor = float(importance_factor.get("importance_factor", 0.988))
        else:
            self.importance_factor = float(importance_factor)

    def __call__(
        self, important_score: float, delta: float
    ) -> Tuple[float, float, float]:
        delta += 1
        new_recency_score = np.exp(-(delta / self.recency_factor))
        new_important_score = important_score * self.importance_factor

        return new_recency_score, new_important_score, delta
