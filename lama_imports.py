from warnings import filterwarnings
filterwarnings('ignore')

import os, re, joblib, tempfile, ctypes
from os import path, walk, getpid
from psutil import Process
from collections import Counter
from itertools import product
from gc import collect

libc = ctypes.CDLL("libc.so.6")

from IPython.display import display_html, clear_output
from pprint import pprint
from functools import partial
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style, init
from tqdm.notebook import tqdm

# Essential DS libraries
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from sklearn.metrics import *

from sklearn.model_selection import PredefinedSplit as PDS
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

# Color printing
def PrintColor(text: str, color = Fore.BLUE, style = Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string"
    print(style + color + text + Style.RESET_ALL)

print(f"---> CUDA available = {torch.cuda.is_available()}\n\n")
