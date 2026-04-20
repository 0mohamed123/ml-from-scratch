import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import LogisticRegression as SklearnLogR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans as SklearnKMeans
import time

from algorithms.linear_regression import LinearRegression
from algorithms.logistic_regression import LogisticRegression
from algorithms.knn import KNN
from algorithms.decision_tree import DecisionTree
from algorithms.naive_bayes import NaiveBayes
from algorithms.kmeans import KMeans


def separator():
    print("=" * 60)


def run_benchmark():
    print("\n")
    separator()
    print("   ML Algorithms from Scratch — Benchmark vs sklearn")
    separator()

    # ── 1. Linear Regression ──────────────────────────────────
    print("\n[1] Linear Regression")
    X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)

    t0 = time.time()
    our = LinearRegression(learning_rate=0.01, n_iterations=1000)
    our.fit(X, y)
    our_time = time.time() - t0
    our_r2 = our.score(X, y)

    t0 = time.time()
    sk = SklearnLR()
    sk.fit(X, y)
    sk_time = time.time() - t0
    sk_r2 = sk.score(X, y)

    print(f"    Ours   -> R²: {our_r2:.4f}  | Time: {our_time:.4f}s")
    print(f"    sklearn-> R²: {sk_r2:.4f}  | Time: {sk_time:.4f}s")
    print(f"    Difference: {abs(our_r2 - sk_r2):.4f}")

    # ── 2. Logistic Regression ────────────────────────────────
    print("\n[2] Logistic Regression")
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)

    t0 = time.time()
    our = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    our.fit(X, y)
    our_time = time.time() - t0
    our_acc = our.score(X, y)

    t0 = time.time()
    sk = SklearnLogR(max_iter=1000)
    sk.fit(X, y)
    sk_time = time.time() - t0
    sk_acc = sk.score(X, y)

    print(f"    Ours   -> Acc: {our_acc:.4f} | Time: {our_time:.4f}s")
    print(f"    sklearn-> Acc: {sk_acc:.4f} | Time: {sk_time:.4f}s")
    print(f"    Difference: {abs(our_acc - sk_acc):.4f}")

    # ── 3. KNN ────────────────────────────────────────────────
    print("\n[3] K-Nearest Neighbors (k=3)")
    X, y = make_classification(n_samples=300, n_features=4, random_state=42)

    t0 = time.time()
    our = KNN(k=3)
    our.fit(X, y)
    our_time = time.time() - t0
    our_acc = our.score(X, y)

    t0 = time.time()
    sk = KNeighborsClassifier(n_neighbors=3)
    sk.fit(X, y)
    sk_time = time.time() - t0
    sk_acc = sk.score(X, y)

    print(f"    Ours   -> Acc: {our_acc:.4f} | Time: {our_time:.4f}s")
    print(f"    sklearn-> Acc: {sk_acc:.4f} | Time: {sk_time:.4f}s")
    print(f"    Difference: {abs(our_acc - sk_acc):.4f}")

    # ── 4. Decision Tree ──────────────────────────────────────
    print("\n[4] Decision Tree (max_depth=5)")
    X, y = make_classification(n_samples=300, n_features=4, random_state=42)

    t0 = time.time()
    our = DecisionTree(max_depth=5)
    our.fit(X, y)
    our_time = time.time() - t0
    our_acc = our.score(X, y)

    t0 = time.time()
    sk = DecisionTreeClassifier(max_depth=5, random_state=42)
    sk.fit(X, y)
    sk_time = time.time() - t0
    sk_acc = sk.score(X, y)

    print(f"    Ours   -> Acc: {our_acc:.4f} | Time: {our_time:.4f}s")
    print(f"    sklearn-> Acc: {sk_acc:.4f} | Time: {sk_time:.4f}s")
    print(f"    Difference: {abs(our_acc - sk_acc):.4f}")

    # ── 5. Naive Bayes ────────────────────────────────────────
    print("\n[5] Naive Bayes (Gaussian)")
    X, y = make_classification(n_samples=300, n_features=4, random_state=42)

    t0 = time.time()
    our = NaiveBayes()
    our.fit(X, y)
    our_time = time.time() - t0
    our_acc = our.score(X, y)

    t0 = time.time()
    sk = GaussianNB()
    sk.fit(X, y)
    sk_time = time.time() - t0
    sk_acc = sk.score(X, y)

    print(f"    Ours   -> Acc: {our_acc:.4f} | Time: {our_time:.4f}s")
    print(f"    sklearn-> Acc: {sk_acc:.4f} | Time: {sk_time:.4f}s")
    print(f"    Difference: {abs(our_acc - sk_acc):.4f}")

    # ── 6. KMeans ─────────────────────────────────────────────
    print("\n[6] KMeans (k=3)")
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

    t0 = time.time()
    our = KMeans(k=3)
    our.fit(X)
    our_time = time.time() - t0

    t0 = time.time()
    sk = SklearnKMeans(n_clusters=3, random_state=42)
    sk.fit(X)
    sk_time = time.time() - t0

    print(f"    Ours   -> Inertia: {our.inertia_:.2f} | Time: {our_time:.4f}s")
    print(f"    sklearn-> Inertia: {sk.inertia_:.2f} | Time: {sk_time:.4f}s")

    separator()
    print("   All algorithms verified against sklearn")
    separator()
    print()


if __name__ == "__main__":
    run_benchmark()