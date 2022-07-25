#ignore warnings
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib as plt

import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk import trigrams
from nltk import bigrams
from collections import Counter
from typing import Dict, List, Optional, Union, cast


import os
import time
import csv


import requests
import prepare

from sklearn.model_selection import train_test_split

from scipy import stats
from scipy.stats import pearsonr, spearmanr


def split_data(df):
    '''
    This function takes in a dataframe and splits it into three subgroups: train, test, validate
    for proper evalution, statistical testing, and modeling. Three dataframes are returned.
    '''
#train, test, split
train_validate, test = train_test_split(df, test_size = .2, random_state = 123, stratify = df.language)
train, validate = train_test_split(train_validate, test_size = .3, random_state = 123, stratify = train_validate.language)
return train, validate, test


   