# Predicting Forest Cover Type

This project aims to use Random Forests to predict the cover type of a forest

### Prerequisites

This uses Python 3 and these libraries:

```
import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```
The dataset is stored as cov_type



## Authors

* **Lucy Hayes** - (https://github.com/luicyfruit)
