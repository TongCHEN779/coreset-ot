"""Evaluation harness: models, DSA augmentation and the train-from-scratch protocol."""
from .augment import dsa, strategy_for
from .models import ARCHS, make_model
from .train import default_epochs, evaluate_coreset, train_and_evaluate

__all__ = ["ARCHS", "make_model", "dsa", "strategy_for", "default_epochs", "train_and_evaluate", "evaluate_coreset"]
