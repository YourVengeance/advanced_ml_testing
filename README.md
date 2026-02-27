# 🧠 Statistical Comparison of Parametric Classification Models

## 📌 Project Description

This project implements and compares parametric classification models on a binary classification problem using rigorous statistical evaluation.

The objective is to:

- Implement classification models based on mathematical foundations  
- Apply proper experimental design techniques  
- Evaluate model generalization using cross-validation  
- Perform statistical hypothesis testing  
- Measure practical significance using effect size  

The focus of this project is **understanding theoretical foundations and statistical validation**, rather than only achieving high accuracy.

---

## 📂 Problem Setting

Binary Classification:

Given a feature vector:

x ∈ ℝⁿ  

Predict class label:

y ∈ {0, 1}

---

## 🤖 Models Implemented

### 1️⃣ Logistic Regression (Discriminative Model)

Logistic Regression directly models the conditional probability:

P(y = 1 | x)

### Model Equation:

\[
z = w^T x + b
\]

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

\[
P(y=1|x) = \sigma(w^T x + b)
\]

Decision Rule:

\[
\hat{y} =
\begin{cases}
1 & \text{if } P(y=1|x) ≥ 0.5 \\
0 & \text{otherwise}
\end{cases}
\]

Optimization is performed using **Gradient Descent**:

\[
w := w - \eta \frac{\partial J}{\partial w}
\]

\[
b := b - \eta \frac{\partial J}{\partial b}
\]

---

### 2️⃣ Linear Discriminant Analysis (Generative Model)

LDA models the class-conditional distribution:

\[
x | y = k \sim \mathcal{N}(\mu_k, \Sigma)
\]

Assumptions:

- Each class follows a Gaussian distribution
- Classes share the same covariance matrix

Using Bayes' theorem:

\[
P(y|x) = \frac{P(x|y)P(y)}{P(x)}
\]

Discriminant function:

\[
g_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log \pi_k
\]

Class with highest \( g_k(x) \) is selected.

---

## 🔬 Experimental Methodology

### 1️⃣ Cross-Validation

Stratified K-Fold Cross Validation is used to estimate generalization performance.

This ensures:
- Proper class balance in each fold
- Reduced variance in performance estimation

---

### 2️⃣ Interval Estimation

Confidence Interval for mean accuracy:

\[
\bar{x} \pm t_{\alpha/2} \frac{s}{\sqrt{n}}
\]

Where:

- \( \bar{x} \) = mean performance
- \( s \) = sample standard deviation
- \( n \) = number of folds
- \( t \) = critical value from t-distribution

---

### 3️⃣ Hypothesis Testing (Paired t-test)

Used to compare two classifiers across folds.

Test statistic:

\[
t = \frac{\bar{d}}{s_d / \sqrt{n}}
\]

Where:

- \( \bar{d} \) = mean difference between models
- \( s_d \) = standard deviation of differences
- \( n \) = number of folds

Null Hypothesis:

H₀: No difference in performance

---

### 4️⃣ Effect Size (Cohen’s d)

Measures practical significance:

\[
d = \frac{\bar{d}}{s_d}
\]

This quantifies how large the difference is relative to variability.

---

## 📈 Evaluation Concepts Used

- Accuracy
- Cross-validation
- Confidence intervals
- Hypothesis testing
- Effect size
- Model selection
- Bias-variance considerations

---

## 🧠 Conceptual Focus

This project emphasizes:

- Discriminative vs Generative modeling
- Parametric classification
- Gaussian assumptions
- Gradient-based optimization
- Statistical comparison of models
- Scientific experimental design

---
## 🚀 How to Run

```bash
pip install numpy scikit-learn scipy matplotlib
python advanced_breast.py
