import sys
import os, glob

import pandas as pd
import numpy as np
import seaborn as sns
import math
import csv

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.image as mpimg

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs

from IPython.display import Image, IFrame, display

home=os.path.expanduser('~')
