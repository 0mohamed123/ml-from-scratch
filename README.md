# ML Algorithms from Scratch

![Tests](https://github.com/0mohamed123/ml-from-scratch/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-only-green)

Implementation of core ML algorithms using only NumPy — no sklearn, no shortcuts.
Each algorithm matches sklearn results verified by 28 automated tests.

## Algorithms Implemented

| Algorithm | Type | Accuracy vs sklearn |
|-----------|------|-------------------|
| Linear Regression | Regression | R2 diff: 0.0000 |
| Logistic Regression | Classification | Acc diff: 0.0000 |
| K-Nearest Neighbors | Classification | Acc diff: 0.0000 |
| Decision Tree | Classification | Acc diff: 0.0000 |
| Naive Bayes | Classification | Acc diff: 0.0000 |
| K-Means | Clustering | Inertia match: exact |

## Quick Start

    git clone https://github.com/0mohamed123/ml-from-scratch.git
    cd ml-from-scratch
    pip install -r requirements.txt
    python -m benchmarks.compare_sklearn
    pytest tests/ -v

## Benchmark Results

    [1] Linear Regression
        Ours   -> R2: 0.9930  | Time: 0.0120s
        sklearn-> R2: 0.9930  | Time: 0.0000s
        Difference: 0.0000

    [2] Logistic Regression
        Ours   -> Acc: 0.8920 | Time: 0.0290s
        sklearn-> Acc: 0.8920 | Time: 0.0031s
        Difference: 0.0000

    [3] K-Nearest Neighbors
        Ours   -> Acc: 0.9800 | Time: 0.0000s
        sklearn-> Acc: 0.9800 | Time: 0.0010s
        Difference: 0.0000

    [4] Decision Tree
        Ours   -> Acc: 0.9867 | Time: 0.1870s
        sklearn-> Acc: 0.9867 | Time: 0.0010s
        Difference: 0.0000

    [5] Naive Bayes
        Ours   -> Acc: 0.9200 | Time: 0.0000s
        sklearn-> Acc: 0.9200 | Time: 0.0000s
        Difference: 0.0000

    [6] KMeans
        Ours   -> Inertia: 566.86 | Time: 0.0050s
        sklearn-> Inertia: 566.86 | Time: 0.4216s

## Key Design Decisions

- Every algorithm follows the same interface: fit(), predict(), score()
- Gradient Descent with configurable learning rate and iterations
- KMeans++ initialization for better convergence
- Decision Tree supports both Gini and Entropy criteria
- All metrics implemented from scratch in utils/metrics.py

## Technologies

- Python 3.10+
- NumPy (core computations)
- pytest (28 tests)
- GitHub Actions (CI/CD)