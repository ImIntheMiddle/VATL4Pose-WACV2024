"""Training of the pose estimator and the whole-body autoencoder.

* :mod:`~vatl4pose.train.finetune` retrains the estimator inside the active-learning loop.
* :mod:`~vatl4pose.train.initial` trains the initial estimator, and
  :mod:`~vatl4pose.train.autoencoder` pretrains the whole-body autoencoder; both produce the
  starting weights of the loop.
"""

from .autoencoder import AETrainConfig, hybrid_features, train_wholebody_ae
from ._common import shuffle_order
from .finetune import Finetuner, RetrainConfig, heatmap_loss
from .initial import InitialTrainConfig, evaluate_estimator, train_estimator

__all__ = [
    "AETrainConfig",
    "Finetuner",
    "InitialTrainConfig",
    "RetrainConfig",
    "evaluate_estimator",
    "heatmap_loss",
    "hybrid_features",
    "shuffle_order",
    "train_estimator",
    "train_wholebody_ae",
]
