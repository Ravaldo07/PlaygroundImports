import numpy as np, pandas as pd
import polars as pl
import polars.selectors as cs
import re, os, joblib, logging
from gc import collect

from IPython.display import display_html, clear_output
from pprint import pprint
from tqdm.notebook import tqdm
from colorama import Fore, Back, Style
from os import path, walk, getpid
from psutil import Process
import ctypes
libc = ctypes.CDLL("libc.so.6")

from warnings import filterwarnings
filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold as SKF, GroupKFold as GKF
from sklearn.metrics import *
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.core.metrics import make_scorer as ag_make_scorer

# Color printing
def PrintColor(text: str, color = Fore.BLUE, style = Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string"
    print(style + color + text + Style.RESET_ALL)
