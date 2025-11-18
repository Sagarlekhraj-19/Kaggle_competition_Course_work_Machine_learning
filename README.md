# 🚗 Enhanced Safe Driver Prediction Challenge

> **Machine Learning Project**: Predicting insurance claim probability using advanced classification techniques

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Models Implemented](#-models-implemented)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🎯 Overview

This project tackles the **Enhanced Safe Driver Prediction Challenge**, focusing on predicting the probability that an auto insurance policyholder will file a claim. Built on an improved version of the Porto Seguro dataset, this project emphasizes:

- ✨ **Smart Feature Engineering**
- ⚖️ **Handling Severely Imbalanced Data** (94.9% vs 5.1%)
- 🎯 **Maximizing AUROC Performance**
- 🔄 **Robust Cross-Validation**

### 🏆 Competition Achievement

**Best Model**: CatBoost with Hyperparameter Tuning
- **Kaggle Public Score**: `0.64138` 🥇
- **Training Time**: 12.5 hours
- **Key Success**: Perfect generalization (CV score matched Kaggle score)

---

## 💡 Problem Statement

Insurance companies must assess risk to determine premiums and minimize financial losses. This project develops a machine learning classifier to:

- 📊 Predict claim probability based on policyholder and vehicle features
- 🎯 Achieve high AUROC for effective risk discrimination
- 💰 Enable personalized premium pricing
- 🚫 Reduce fraudulent claims

---

## 📊 Dataset

### Statistics

| Metric | Value |
|--------|-------|
| **Training Samples** | 296,209 |
| **Test Samples** | 126,948 |
| **Features** | 67 |
| **Numeric Variables** | 37 |
| **Categorical Variables** | 30 |
| **Class Imbalance** | 18.5:1 |

### Feature Categories

```
📍 Individual Variables (ps_ind_*)
🚙 Car-Related Variables (ps_car_*)
🗺️ Regional Variables (ps_reg_*)
🧮 Calculated Variables (ps_calc_*)
⚙️ Engineered Features (feature1-8)
🎯 Target Variable (binary: 0/1)
```

### Data Quality Challenges

- ⚠️ **Missing Data**: Up to 69% in some features
- ⚖️ **Severe Class Imbalance**: 94.9% non-claims
- 🔗 **High Correlation**: 21 variables flagged
- 0️⃣ **Zero-Inflation**: Multiple variables

---

## 📁 Project Structure

```
enhanced-safe-driver-prediction/
│
├── 📓 kaggle.ipynb                 # Main training notebook
├── 📄 kaggle_report.pdf           # Comprehensive project report
│
├── 📊 Data/
│   ├── train1.csv                 # Training dataset
│   └── test.csv                   # Test dataset
│
├── 💾 Models/
│   ├── submission_CatBoost.csv    # Winner! 🏆
│   ├── submission_RandomForest.csv
│   ├── submission_AdaBoost.csv
│   ├── submission_DecisionTree.csv
│   ├── submission_KNN.csv
│   └── submission_NaiveBayes.csv
│
├── 📈 Visualizations/
│   └── model_training_comparison.png
│
└── 📖 README.md                   # This file
```

---

## 🤖 Models Implemented

### 1. Categorical Naive Bayes
- ⚡ **Training Time**: 1.86s
- 📊 **Train AUROC**: 0.6423
- 🎯 **Kaggle Score**: Not submitted
- 💭 **Note**: Fast baseline with independence assumption

### 2. K-Nearest Neighbors (k=5)
- ⚡ **Training Time**: 4.21s
- 📊 **Train AUROC**: 0.9240 (Highest!)
- 🎯 **Kaggle Score**: 0.50623 (Worst - Overfitting!)
- ⚠️ **Warning**: Memorized training data

### 3. Decision Tree (depth=10)
- ⚡ **Training Time**: 12.50s
- 📊 **Train AUROC**: 0.6743
- 🎯 **Kaggle Score**: 0.57333
- 📋 **Nodes**: 1,023 | **Leaves**: 512

### 4. Random Forest (100 trees)
- ⚡ **Training Time**: 48.85s
- 📊 **Train AUROC**: 0.9116
- 🎯 **Kaggle Score**: 0.59801
- ⚠️ **Issue**: 34% performance drop (overfitting)

### 5. AdaBoost (100 estimators)
- ⚡ **Training Time**: 341.44s
- 📊 **Train AUROC**: 0.6438
- 🎯 **Kaggle Score**: 0.63016 (3rd place)
- 💡 **Strength**: Good with imbalanced data

### 6. CatBoost (Grid Search) 🏆
- ⚡ **Training Time**: 45,005s (12.5 hours)
- 📊 **Train AUROC**: 0.6383 (CV)
- 🎯 **Kaggle Score**: 0.64138 (BEST!)
- 🎨 **Parameters**: 243 combinations × 3 folds = 729 fits
- ✨ **Key**: Perfect generalization (CV matched Kaggle)

#### Optimal Hyperparameters
```python
{
    'iterations': 500,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 5,
    'border_count': 32,
    'class_weights': [1, 5]
}
```

---

## 📈 Results

### Final Kaggle Leaderboard

| Rank | Model | Kaggle Score | Training AUROC | Gap |
|------|-------|--------------|----------------|-----|
| 🥇 1 | **CatBoost** | **0.64138** | 0.6383 | **+0.5%** ✅ |
| 🥈 2 | CatBoost v2 | 0.63825 | 0.6383 | ±0.0% |
| 🥉 3 | AdaBoost | 0.63016 | 0.6438 | -2.1% |
| 4 | Decision Tree | 0.57333 | 0.6743 | -15.0% |
| 5 | Random Forest | 0.59801 | 0.9116 | **-34.4%** ⚠️ |
| 6 | KNN | 0.50623 | 0.9240 | **-45.2%** 🚫 |

### Performance Visualization

![Model Comparison](Visualizations/model_training_comparison.png)

### Top 10 Important Features (CatBoost)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `ps_ind_03` | 9.3370 | Individual |
| 2 | `ps_car_13` | 7.1361 | Car |
| 3 | `ps_reg_01` | 4.9687 | Regional |
| 4 | `ps_ind_15` | 4.6452 | Individual |
| 5 | `ps_reg_02` | 3.6249 | Regional |
| 6 | `ps_ind_05_cat_0.0` | 3.5811 | Categorical |
| 7 | `ps_ind_17_bin` | 3.3785 | Binary |
| 8 | `ps_reg_03` | 3.1532 | Regional |
| 9 | `feature4` | 2.5495 | Engineered |
| 10 | `ps_car_14` | 2.4677 | Car |

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.10+
Jupyter Notebook
```

### Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install catboost xgboost lightgbm
pip install jupyter notebook
```

### Clone Repository

```bash
git clone https://github.com/yourusername/safe-driver-prediction.git
cd safe-driver-prediction
```

---

## 💻 Usage

### 1. Data Preparation

```python
# Load data
import pandas as pd
train = pd.read_csv('train1.csv')
test = pd.read_csv('test.csv')

# Check dimensions
print(f"Training: {train.shape}")
print(f"Testing: {test.shape}")
```

### 2. Run Training Pipeline

```bash
jupyter notebook kaggle.ipynb
```

### 3. Generate Predictions

All models generate submission files:
```
submission_CatBoost.csv      # Best model
submission_RandomForest.csv
submission_AdaBoost.csv
submission_DecisionTree.csv
submission_KNN.csv
```

### 4. Submit to Kaggle

```bash
kaggle competitions submit -c [competition-name] -f submission_CatBoost.csv -m "CatBoost submission"
```

---

## 🔑 Key Findings

### 🎯 Critical Lessons

#### 1. **Training Scores Are Deceptive**
```
KNN: 0.924 training → 0.506 Kaggle (-45% drop!)
CatBoost: 0.638 CV → 0.641 Kaggle (+0.5% gain!)
```
**Lesson**: Never trust training metrics without proper cross-validation.

#### 2. **Conservative Parameters Win**
Initially criticized settings proved optimal:
- `l2_leaf_reg=5` (max regularization)
- `learning_rate=0.03` (slow learning)
- Prevented overfitting that destroyed Random Forest

#### 3. **Time Investment Pays Off**
- 12.5 hours training → 1st place
- 729 model fits prevented overfitting
- Thoroughness beats speed in competitions

#### 4. **Class Imbalance Handling**
```python
class_weights = [1, 5]  # 5x weight for minority class
```
Essential for AUROC performance on imbalanced data.

---

## 🧮 Data Preprocessing Pipeline

```python
# 1. Handle Missing Values
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='mean')

# 2. Drop High-Missing Features
drop_cols = ['ps_car_03_cat', 'ps_car_05_cat']  # 69%, 45% missing

# 3. One-Hot Encoding
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[cat_cols])

# 4. Feature Scaling (for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# 5. Final Dataset
X_final = pd.concat([X_encoded, X_numeric, X_binary], axis=1)
```

---

## 📊 Model Evaluation Metrics

### Primary Metric: AUROC
```python
from sklearn.metrics import roc_auc_score

auroc = roc_auc_score(y_true, y_pred_proba)
```

### Why AUROC?
- ✅ Handles class imbalance
- ✅ Threshold-independent
- ✅ Measures discrimination ability
- ❌ Accuracy misleading (94% by predicting all zeros!)

---

## 🎓 Technical Highlights

### Grid Search Configuration
```python
param_grid = {
    'iterations': [300, 500, 700],
    'learning_rate': [0.03, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64, 128]
}

# 3×3×3×3×3 = 243 combinations
# 243 × 3 folds = 729 model fits
```

### Cross-Validation Strategy
```python
GridSearchCV(
    estimator=catboost_model,
    param_grid=param_grid,
    cv=3,  # 3-fold CV
    scoring='roc_auc',
    n_jobs=-1
)
```

---

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

## 👨‍💻 Author

**Sagar Lekhraj**
- 🎓 ERP: 29325
- 🏫 Institution: IBA Karachi
- 📧 Email: [your-email@example.com]
- 🔗 LinkedIn: [Your LinkedIn Profile]
- 💻 GitHub: [@yourusername](https://github.com/yourusername)

**Course**: CSE 472 - Introduction to Machine Learning  
**Instructor**: Dr. Sajjad Haider, PhD  
**Department**: Computer Science

---

## 📚 References

1. Porto Seguro Safe Driver Prediction Dataset
2. Scikit-learn Documentation
3. CatBoost Official Documentation
4. Kaggle Competition Guidelines

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- IBA Karachi Computer Science Department
- Dr. Sajjad Haider for course guidance
- Kaggle community for inspiration
- CatBoost team for excellent documentation

---

<div align="center">

### ⭐ If you found this project helpful, please star the repository! ⭐

**Made with ❤️ and ☕ by Sagar Lekhraj**

</div>

---

## 📅 Project Timeline

```
Week 1: Data Exploration & EDA
Week 2: Preprocessing & Feature Engineering
Week 3: Baseline Models (Naive Bayes, KNN, Decision Tree)
Week 4: Ensemble Methods (Random Forest, AdaBoost)
Week 5: CatBoost Hyperparameter Tuning (12.5 hours!)
Week 6: Final Submission & Report
```

---

## 🔮 Future Work

- [ ] Implement SMOTE for better class balance
- [ ] Try XGBoost and LightGBM
- [ ] Deep learning approaches (Neural Networks)
- [ ] Ensemble stacking of top models
- [ ] Feature selection optimization
- [ ] Advanced feature engineering
- [ ] Bayesian optimization for hyperparameters

---

**Last Updated**: November 2024
