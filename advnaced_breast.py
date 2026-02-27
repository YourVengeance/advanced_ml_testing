# ============================================
# ADVANCED BREAST CANCER MODEL COMPARISON
# ============================================

import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import ttest_rel, t

# ============================================
# LOAD DATA
# ============================================

data = load_breast_cancer()
X = data.data
y = data.target

# ============================================
# LOGISTIC REGRESSION FROM SCRATCH
# ============================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for _ in range(epochs):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)

        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)

        w -= lr * dw
        b -= lr * db

    return w, b

def predict_logistic(X, w, b):
    z = np.dot(X, w) + b
    probs = sigmoid(z)
    return (probs >= 0.5).astype(int), probs


# ============================================
# LDA FROM SCRATCH
# ============================================

def train_lda(X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]

    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)

    cov = np.cov(X.T)

    return mu0, mu1, cov

def predict_lda(X, mu0, mu1, cov):
    inv_cov = np.linalg.pinv(cov)

    def discriminant(x, mu):
        return np.dot(np.dot(x, inv_cov), mu) - 0.5 * np.dot(np.dot(mu, inv_cov), mu)

    preds = []
    scores = []

    for x in X:
        g0 = discriminant(x, mu0)
        g1 = discriminant(x, mu1)
        preds.append(1 if g1 > g0 else 0)
        scores.append(g1 - g0)

    return np.array(preds), np.array(scores)


# ============================================
# EXPERIMENT SETTINGS
# ============================================

n_splits = 20
n_repeats = 3

results = {
    "Logistic": [],
    "LDA": [],
    "SVM": [],
    "KNN": []
}

runtimes = {
    "Logistic": [],
    "LDA": [],
    "SVM": [],
    "KNN": []
}

roc_data = {}

# ============================================
# REPEATED STRATIFIED 20-FOLD CV
# ============================================

for repeat in range(n_repeats):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42+repeat)

    for train_idx, test_idx in skf.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ================= Logistic =================
        start = time.time()
        w, b = train_logistic(X_train, y_train)
        y_pred, y_prob = predict_logistic(X_test, w, b)
        runtimes["Logistic"].append(time.time() - start)

        results["Logistic"].append(accuracy_score(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data["Logistic"] = (fpr, tpr)

        # ================= LDA =================
        start = time.time()
        mu0, mu1, cov = train_lda(X_train, y_train)
        y_pred, scores = predict_lda(X_test, mu0, mu1, cov)
        runtimes["LDA"].append(time.time() - start)

        results["LDA"].append(accuracy_score(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_data["LDA"] = (fpr, tpr)

        # ================= SVM =================
        start = time.time()
        svm = SVC(probability=True)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        y_prob = svm.predict_proba(X_test)[:,1]
        runtimes["SVM"].append(time.time() - start)

        results["SVM"].append(accuracy_score(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data["SVM"] = (fpr, tpr)

        # ================= KNN =================
        start = time.time()
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        y_prob = knn.predict_proba(X_test)[:,1]
        runtimes["KNN"].append(time.time() - start)

        results["KNN"].append(accuracy_score(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data["KNN"] = (fpr, tpr)


# ============================================
# ANALYSIS
# ============================================

print("\n========== MODEL SUMMARY ==========\n")

for model in results:
    mean_acc = np.mean(results[model])
    std_acc = np.std(results[model])
    mean_runtime = np.mean(runtimes[model])

    print(f"{model}")
    print("Mean Accuracy:", round(mean_acc, 3))
    print("Std Dev:", round(std_acc, 3))
    print("Mean Runtime (sec):", round(mean_runtime, 4))
    print("-"*40)

# ============================================
# COHEN'S D (Effect Size) Logistic vs LDA
# ============================================

diff = np.array(results["Logistic"]) - np.array(results["LDA"])
cohens_d = np.mean(diff) / np.std(diff, ddof=1)

print("\nCohen's d (Logistic vs LDA):", round(cohens_d, 3))

# ============================================
# PAIRED T-TEST Logistic vs LDA
# ============================================

t_stat, p_value = ttest_rel(results["Logistic"], results["LDA"])
print("Paired t-test p-value:", round(p_value, 4))

# ============================================
# PLOT ROC CURVES
# ============================================

plt.figure(figsize=(8,6))

for model in roc_data:
    fpr, tpr = roc_data[model]
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model} (AUC = {roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()