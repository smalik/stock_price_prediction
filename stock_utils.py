import os
import time
from datetime import datetime, date
from copy import deepcopy
import random
import ssl
from collections import deque

import numpy as np
import pandas as pd


def get_slope(data, indicator: str = 'adjclose'):
    return pd.Series(np.gradient(data[indicator].values), data.index, name=f'slope_{indicator}')
