# CineMatch: A Movie Recommender System Based on Linear Algebra

## Introduction

CineMatch is a conceptual movie recommender system, designed to predict movies that a user might like based on their past ratings. Unlike many contemporary systems that rely on complex, black-box machine learning libraries, CineMatch is built from the ground up using fundamental tools and concepts from linear algebra.

The primary goal of this project is to provide a clear and tangible demonstration of how mathematical principles, specifically matrix factorization, can be applied to solve real-world problems like building a personalized recommendation engine.

## Core Concept: Matrix Factorization

The system is based on **Matrix Factorization with Biases**. The core idea is to approximate the matrix of user-item ratings as the product of two lower-rank matrices: a user-factor matrix ($P$) and an item-factor matrix ($Q$). Each user $u$ is represented by a vector $P_u$, and each item $i$ is represented by a vector $Q_i$.

The predicted rating $\hat{r}_{ui}$ for a user $u$ on an item $i$ is modeled as:

$$\hat{r}_{ui} = \mu + b_u + b_i + P_u^\top Q_i$$

Where:
- $\mu$: The global average rating.
- $b_u$: The bias term for user $u$ (e.g., some users give higher ratings in general).
- $b_i$: The bias term for item $i$ (e.g., some movies are generally better received).
- $P_u^\top Q_i$: The dot product of the user and item latent factor vectors, capturing the interaction between them.

The model is trained by minimizing the regularized squared error over the set of observed ratings ($\Omega$) using **Stochastic Gradient Descent (SGD)**:

$$\min_{P,Q,b_u,b_i} \sum_{(u,i)\in\Omega} \big(r_{ui} - \hat{r}_{ui}\big)^2 + \lambda\Big(\lVert P\rVert^2 + \lVert Q\rVert^2 + \lVert b_u\rVert^2 + \lVert b_i\rVert^2\Big)$$

## Features

* **Data Loading & Mapping:** Ingests CSV rating data and maps user/item IDs to contiguous integer indices.
* **Chronological Data Splitting:** Implements a per-user, time-aware split into training, validation, and test sets to prevent data leakage.
* **Baseline Models:** Includes a global mean and a regularized bias model for performance comparison.
* **Matrix Factorization via SGD:** A from-scratch implementation of SGD with early stopping to train the model.
* **Hyperparameter Tuning:** A loop to find the optimal rank ($k$) for the factorization based on validation RMSE.
* **Latent Factor Interpretation:** Analysis of the learned item factors to uncover underlying themes or genres.
* **Top-N Recommendation:** Generates personalized movie recommendations for any given user.

## How It Works

1.  **Load Data:** The `ranking_test.csv` file is loaded into a pandas DataFrame.
2.  **Preprocess:** The dataset is cleaned by removing users and items with too few ratings to ensure a certain density. User and item IDs are then mapped to internal integer indices.
3.  **Split Data:** The ratings for each user are split chronologically into training, validation, and test sets. This ensures we only use past ratings to predict future ones.
4.  **Train Model:** The Matrix Factorization model is trained on the training data using SGD. The validation set is used to monitor for overfitting and to implement early stopping.
5.  **Tune Hyperparameters:** The training process is repeated for a range of latent factor dimensions (rank $k$). The model that performs best on the validation set (lowest RMSE) is selected.
6.  **Evaluate:** The final, tuned model is evaluated once on the unseen test set to report its generalization performance.
7.  **Recommend:** For a given user, the model predicts ratings for all movies they haven't seen and returns the top-N highest-rated ones.

## Getting Started

### Prerequisites

Make sure you have the following Python libraries installed:

* `numpy`
* `pandas`
* `scipy`
* `matplotlib`
* `scikit-learn` (for clustering and PCA)

### Running the Code

1.  Place your ratings data in a CSV file named `ranking_test.csv` in the same directory as the notebook (one already exists). The CSV should contain four columns: `user_id`, `item_id`, `rating` and `title`.
2.  Open and run the `cinematch.ipynb` notebook in a Jupyter environment. The notebook will execute all steps from data loading to generating recommendations.
