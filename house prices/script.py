import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso

train_df = pd.read_csv("train.csv", low_memory=False)