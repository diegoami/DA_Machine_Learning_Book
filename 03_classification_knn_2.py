# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import numpy
numpy.set_printoptions(threshold=numpy.nan)
np.set_printoptions(threshold=np.inf)
# to make this notebook's output stable across runs
np.random.seed(42)


import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

X, y = shuffle(mnist["data"], mnist["target"])


#X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
X_train, X_test, y_train, y_test = X[:8000], X[8000:10000], y[:8000], y[8000:10000]
print(X_train.shape)
from scipy.ndimage.interpolation import shift
def shift_digit(digit_array, dx, dy, new=0):
    #print(digit_array, digit_array.shape)
    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)


from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_, grid_search.best_score_)