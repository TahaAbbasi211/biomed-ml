"""Stateful preprocessing. These are fitted on the training part of every
cross-validation fold (never on the test part), so they cannot leak."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Column positions follow datasets.heart.FEATURE_COLUMNS.
HEART_NUM_IDX = [0, 3, 4, 7, 9]               # age, trestbps, chol, thalach, oldpeak
HEART_CAT_IDX = [1, 2, 5, 6, 8, 10, 11, 12]   # sex, cp, fbs, restecg, exang, slope, ca, thal


def heart_tabular():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, HEART_NUM_IDX),
                              ("cat", categorical, HEART_CAT_IDX)])


# To add a preprocessor: write a function that returns an unfitted transformer
# and register it here.
PREPROCESSORS = {"heart_tabular": heart_tabular}
