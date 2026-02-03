# Online News Popularity Prediction using PySpark

A big data classification project using PySpark MLlib to predict the popularity of online news articles based on their features.

## Project Overview

This project implements **6 classification algorithms** to predict whether an online news article will be popular (>1,400 shares) using the Online News Popularity dataset from UCI Machine Learning Repository.

### Methodology

Following the approach from the original research paper ([Fernandes et al., 2015](https://www.semanticscholar.org/paper/A-Proactive-Intelligent-Decision-Support-System-for-Fernandes-Vinagre/ad7f3da7a5d6a1e18cc5a176f18f52687b912fea)):

- **Binary Classification**: Articles with >1,400 shares = Popular (1), otherwise Unpopular (0)
- **Rolling Window Validation**: Window size (W) = 10,000, Step size (L) = 1,000
- **Train/Validation Split**: 70% training, 30% validation within each window
- **Feature Engineering**: 58 predictive features with standardization (mean=0, std=1)
- **Time-Series Approach**: Data sorted chronologically by `timedelta` for temporal validity

## Dataset

**Source**: [Online News Popularity Dataset](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)

- **Articles**: 39,644 articles from Mashable
- **Features**: 61 attributes (58 predictive + 2 non-predictive + 1 target)
- **Target Variable**: `shares` (number of social media shares)

## Algorithms Implemented

### Part 1: Basic Classifiers (`big_data_3cfu.ipynb`)

1. **Logistic Regression** 
2. **Naive Bayes** (Gaussian) 
3. **Decision Tree** 

### Part 2: Advanced Classifiers (`big_data_6cfu.ipynb`)

4. **Random Forest** 
5. **Gradient-Boosted Tree** 
6. **Linear SVM** 

## Installation

### Prerequisites

- Python 3.11
- Java 21

### Setup

1. **Clone or navigate to the project directory**:

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync
   ```

## Usage

## Evaluation Metrics

All models are evaluated using:

1. **AUC-ROC**
2. **Accuracy**
3. **F1-Score**
4. **Precision**
5. **Recall**