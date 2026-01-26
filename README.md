# Online News Popularity Prediction using PySpark

A big data classification project using PySpark MLlib to predict the popularity of online news articles based on their features.

## Project Overview

This project implements **6 classification algorithms** to predict whether an online news article will be popular (>1,400 shares) using the Online News Popularity dataset from UCI Machine Learning Repository.

### Methodology

Following the approach from the original research paper ([Fernandes et al., 2015](https://www.semanticscholar.org/paper/A-Proactive-Intelligent-Decision-Support-System-for-Fernandes-Vinagre/ad7f3da7a5d6a1e18cc5a176f18f52687b912fea)):

- **Binary Classification**: Articles with >1,400 shares = Popular (1), otherwise Unpopular (0)
- **Rolling Window Validation**: Window size (W) = 10,000, Step size (L) = 1,000
- **Train/Test Split**: 70% training, 30% testing within each window
- **Feature Engineering**: 58 predictive features with standardization (mean=0, std=1)
- **Time-Series Approach**: Data sorted chronologically by `timedelta` for temporal validity

## Dataset

**Source**: [Online News Popularity Dataset](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)

- **Articles**: 39,644 articles from Mashable
- **Features**: 61 attributes (58 predictive + 2 non-predictive + 1 target)
- **Target Variable**: `shares` (number of social media shares)

### Feature Categories

1. **Content Features**: Tokens, links, images, videos, keywords
2. **NLP Features**: LDA topics, sentiment polarity/subjectivity
3. **Metadata**: Publication channel, day of week, weekend indicator
4. **Engagement**: Shares (target variable)

## Algorithms Implemented

### Part 1: Basic Classifiers (`big_data_3cfu.ipynb`)

1. **Logistic Regression** - Linear baseline model
2. **Naive Bayes** (Gaussian) - Probabilistic classifier
3. **Decision Tree** - Interpretable tree-based model

### Part 2: Advanced Classifiers (`big_data_6cfu.ipynb`)

4. **Random Forest** - Ensemble of 100 decision trees
5. **Gradient-Boosted Tree** - Sequential boosting ensemble
6. **Linear SVM** - Support Vector Machine classifier

## Project Structure

```
.
├── OnlineNewsPopularity.csv       # Dataset (39,644 articles)
├── OnlineNewsPopularity.names     # Feature descriptions
├── introductory_paper.pdf         # Original research paper
├── big_data_3cfu.ipynb           # Part 1: Basic classifiers (3 algorithms)
├── big_data_6cfu.ipynb           # Part 2: Advanced classifiers (3 algorithms)
├── README.md                      # This file
├── pyproject.toml                 # Python dependencies
└── uv.lock                        # Dependency lock file
```

## Installation

### Prerequisites

- Python 3.8+
- Apache Spark 3.x
- Java 8 or 11 (for Spark)

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd /Users/habib/Projects/BIG_DATA/Habib/project
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install pyspark pandas numpy matplotlib scikit-learn jupyter
   ```

3. **Verify Spark installation**:
   ```bash
   pyspark --version
   ```

## Usage

### Running the Notebooks

#### Part 1: Basic Classifiers

```bash
jupyter notebook big_data_3cfu.ipynb
```

This notebook implements:
- Data loading and preprocessing
- Rolling window time-series split
- Logistic Regression, Naive Bayes, Decision Tree
- Model evaluation and comparison
- Feature importance analysis (Decision Tree)
- ROC curves visualization

**Expected Runtime**: 15-30 minutes (depends on hardware)

#### Part 2: Advanced Classifiers

```bash
jupyter notebook big_data_6cfu.ipynb
```

This notebook implements:
- Data preprocessing (same pipeline as Part 1)
- Random Forest, Gradient-Boosted Tree, Linear SVM
- Comprehensive model evaluation
- Feature importance analysis (RF & GBT)
- Final comparison of all 6 algorithms

**Expected Runtime**: 30-60 minutes (ensemble methods are computationally intensive)

### Sequential Execution

Run the notebooks in order to build a complete understanding:

1. **First**: Execute all cells in `big_data_3cfu.ipynb`
2. **Then**: Execute all cells in `big_data_6cfu.ipynb`

Both notebooks are **self-contained** and can be run independently.

## Evaluation Metrics

All models are evaluated using:

1. **AUC-ROC** (Area Under ROC Curve) - Primary metric
2. **Accuracy** - Overall classification accuracy
3. **F1-Score** - Harmonic mean of precision and recall
4. **Precision** - True positive rate
5. **Recall** - Sensitivity

**Note**: Linear SVM doesn't provide probability estimates, so AUC is not available for this model.

## Expected Results

Based on the original paper, anticipated performance:

| Algorithm | Expected AUC | Characteristics |
|-----------|--------------|-----------------|
| Random Forest | ~0.72-0.75 | Best performer, robust |
| Gradient-Boosted Tree | ~0.71-0.74 | Close second, sequential learning |
| Logistic Regression | ~0.65-0.68 | Good baseline |
| Decision Tree | ~0.63-0.67 | Interpretable |
| Naive Bayes | ~0.60-0.63 | Fast, assumes independence |
| Linear SVM | N/A (no prob.) | Accuracy ~0.65-0.68 |

## Technical Details

### Rolling Window Implementation

```python
# Parameters
W = 10000  # Window size
L = 1000   # Step size (using L*3 for efficiency)

# Creates ~10-11 temporal windows
# Each window: 7,000 train samples, 3,000 test samples
```

### Feature Engineering Pipeline

```
Raw Data → Label Creation → Column Cleaning → VectorAssembler → StandardScaler → Models
```

### Spark Configuration

```python
SparkSession.builder \
    .appName("NewsPopularityClassification") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
```

## Key Findings

1. **Ensemble methods** (Random Forest, GBT) consistently outperform single models
2. **Feature importance**: Content features (tokens, links) and temporal features are most predictive
3. **Rolling window validation** ensures models generalize to future articles
4. **Class imbalance**: Dataset is relatively balanced (~50/50 popular/unpopular)

## References

- **Paper**: Fernandes, K., Vinagre, P., & Cortez, P. (2015). *A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News*. Portuguese Conference on Artificial Intelligence.
- **Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
- **Framework**: [Apache Spark MLlib](https://spark.apache.org/mllib/)

## Notes

- This project focuses on **prediction/classification only** (not optimization)
- Time-series validation preserves temporal ordering of articles
- Hyperparameter tuning is not included (focus is on implementation)
- Both notebooks include detailed comments and markdown explanations

## Troubleshooting

### Common Issues

1. **OutOfMemoryError**:
   ```python
   # Increase driver memory in Spark config
   .config("spark.driver.memory", "8g")
   ```

2. **Java not found**:
   ```bash
   # Set JAVA_HOME environment variable
   export JAVA_HOME=/path/to/java
   ```

3. **Slow execution**:
   - Reduce number of rolling windows (increase step size)
   - Use smaller subset of data for testing
   - Enable Spark local mode with more cores: `.master("local[*]")`

## License

This project is for educational purposes. Dataset is from UCI ML Repository (public domain).

## Contact

For questions or issues, please refer to the course materials or instructor.

---

**Project Type**: Big Data Classification with PySpark  
**Course**: Big Data Analytics (3 CFU + 6 CFU)  
**Framework**: Apache Spark 3.x + MLlib  
**Date**: January 2026
