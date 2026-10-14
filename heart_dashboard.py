"""חלק ב - רגרסיה לוגיסטית לניבוי סיכון להתקף לב, ויצוא דשבורד HTML עצמאי.

הסקריפט מאמן את המודל (80% אימון / 20% מבחן, אקראי עם random_state קבוע),
מחשב Precision / Recall / F1 / ROC-AUC על סט המבחן, וכותב את המקדמים,
פרמטרי הנרמול ותוצאות המבחן לתוך heart_dashboard.html.
החישוב של הסיכון למטופל חדש מתבצע בדפדפן עם אותם מקדמים, ולכן הדף עובד בלי שרת.

הרצה:  python heart_dashboard.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
SEED = 42
FEATURES = ["age", "sex", "exang", "ca", "cp", "trtbps", "chol", "fbs", "rest_ecg", "thalach"]
TEMPLATE = BASE / "heart_dashboard_template.html"
OUTPUT = BASE / "heart_dashboard.html"


def load_data():
    df = pd.read_csv(BASE / "heart_data.csv")
    # ca מוגדר 0-3; הערך 4 הוא שגיאה ידועה בנתונים -> מוסר
    return df[df["ca"] <= 3].reset_index(drop=True)


def encode(df):
    """cp ו-rest_ecg הם קטגוריות נומינליות -> One-Hot. sex, exang, fbs בינאריים; ca ספירה (0-3)."""
    X = df[FEATURES].copy()
    X["cp"] = pd.Categorical(X["cp"], categories=[0, 1, 2, 3])
    X["rest_ecg"] = pd.Categorical(X["rest_ecg"], categories=[0, 1, 2])
    return pd.get_dummies(X, columns=["cp", "rest_ecg"], drop_first=True, dtype=int)


def main():
    df = load_data()
    X, y = encode(df), df["target"]

    # 80% אימון / 20% מבחן, נבחרים אקראית (stratify שומר על יחס החולים בשני הסטים)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)

    proba_test = model.predict_proba(scaler.transform(X_test))[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    proba_train = model.predict_proba(scaler.transform(X_train))[:, 1]

    metrics = {
        "precision": precision_score(y_test, pred_test),
        "recall": recall_score(y_test, pred_test),
        "f1": f1_score(y_test, pred_test),
        "roc_auc": roc_auc_score(y_test, proba_test),
        "accuracy": accuracy_score(y_test, pred_test),
        "train_roc_auc": roc_auc_score(y_train, proba_train),
    }
    # בדיקת יציבות: ROC-AUC ב-5 קיפולים על כל הנתונים
    cv = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                         X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")
    metrics["cv_auc_mean"], metrics["cv_auc_std"] = cv.mean(), cv.std()

    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
    fpr, tpr, _ = roc_curve(y_test, proba_test)

    payload = {
        "n_total": len(df), "n_train": len(X_train), "n_test": len(X_test),
        "n_removed": int((pd.read_csv(BASE / "heart_data.csv")["ca"] > 3).sum()),
        "positive_rate": float(y.mean()),
        "columns": X.columns.tolist(),
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)],
        "test": [[int(t), round(float(p), 4)] for t, p in zip(y_test, proba_test)],
        "ranges": {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())]
                   for c in ["age", "trtbps", "chol", "thalach"]},
    }

    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__MODEL__*/{}", json.dumps(payload))
    OUTPUT.write_text(html, encoding="utf-8")
    # עותק בלי שלד המסמך, לפרסום כ-Artifact (נתיב יעד אופציונלי בארגומנט)
    if len(sys.argv) > 1:
        body = html.split("<!--PAGE-START-->", 1)[1].split("<!--PAGE-END-->", 1)[0]
        Path(sys.argv[1]).write_text(body, encoding="utf-8")

    print(f"train={len(X_train)} test={len(X_test)}")
    for k, v in payload["metrics"].items():
        print(f"{k:>14}: {v:.3f}")
    print("confusion:", payload["confusion"])
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
