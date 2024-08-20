# ../src/base.py

import os
import pandas as pd

from src.constants import Constant as c


class BaseEval(object):
    """
    Base evaluator class
    """
    def __init__(self):
        self.save_dir = c.RUN_DIR
        self.dataset = None
        self.records = None
    
    def load_data(self, *, path: str | os.PathLike) -> None:
        """
        Load generation dataset
        """
        raise NotImplementedError("Subclasses should implement this method")

    def run_eval(self) -> None:
        """
        Run evaluation on dataset
        """
        raise NotImplementedError("Subclasses should implement this method")
    
    def get_report(self) -> pd.DataFrame:
        """
        Calculate scores from evaluation
        """
        raise NotImplementedError("Subclasses should implement this method")

    def save_records(self, *, path: str | os.PathLike) -> None:
        """
        Save records as a JSON Lines file
        """
        raise NotImplementedError("Subclasses should implement this method")
