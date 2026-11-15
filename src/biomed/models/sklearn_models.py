from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Contract: every model has fit(X, y), predict(X) and predict_proba(X).
MODELS = {
    "logreg": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    ),
    "svm": lambda: make_pipeline(
        StandardScaler(),
        SVC(probability=True, class_weight="balanced", random_state=42),
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
    ),
    "hist_gb": lambda: HistGradientBoostingClassifier(random_state=42),
}


def build_model(name):
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    return MODELS[name]()
