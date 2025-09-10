# SMS-Spam
This project explores building effective machine learning classifiers to distinguish between SMS spam and ham (non-spam) messages. The lab emphasizes text preprocessing, feature engineering, and model evaluation to understand how different parameters and algorithms affect performance.


# 📱 SMS Spam Classification Lab

## 📖 Overview

This project explores building machine learning classifiers to distinguish between SMS spam and ham (non-spam) messages.

-   **Focus:** text preprocessing, feature engineering, and model evaluation.
-   **Dataset sources:**
    -   Kaggle – SMS Spam Collection Dataset
    -   Cleaned CSV version

---

## 🎯 Learning Goals

-   Understand text classification workflows.
-   Explore preprocessing techniques for text vectorization.
-   Train and evaluate multiple classification models.
-   Handle class imbalance in real-world data.
-   Interpret evaluation metrics: Accuracy, AUC, F1, Precision, Recall, Confusion Matrix.

---

## 📊 Dataset

-   **Size:** ~5572 SMS messages
-   **Class Distribution:** ~13% spam, ~87% ham
-   **Fields:**
    -   `text` – SMS message content
    -   `class` – `spam` or `ham`

---

## ⚙️ Experiment Setup

#### Algorithms
-   Multinomial Naïve Bayes (required)
-   Complement Naïve Bayes
-   Support Vector Machine (SVM)
-   Logistic Regression

#### Vectorization
-   Word Count
-   TF-IDF

#### Preprocessing Variations
-   **`min_df`**: `{1, 3, 5}`
-   **`ngrams`**: `(1,1)`, `(1,2)`, `(1,4)`
-   **`max_features`**: `500`, `1000`, `2000`
-   **`lower`**: `True`

#### Class Imbalance Strategies
-   Unbalanced (baseline)
-   SMOTE
-   Random Oversampling

---

## 📈 Results

### Experiment Outcomes

| Exp | Model | Balancing | TF-IDF | Accuracy | F1 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | MultinomialNB | Unbalanced | ✅ | 0.9684 | 0.8750 | 0.9565 | 0.8063 |
| 2 | MultinomialNB | Unbalanced | ❌ | 0.9770 | 0.9140 | 0.9392 | 0.8901 |
| 3 | MultinomialNB | Unbalanced | ✅ | 0.9785 | 0.9148 | 1.0000 | 0.8429 |
| 4 | ComplementNB | Unbalanced | ✅ | 0.9454 | 0.8233 | 0.7406 | 0.9267 |
| 5 | SVM | Unbalanced | ✅ | 0.9756 | 0.9056 | 0.9645 | 0.8534 |
| 6 | Logistic Regression | Unbalanced | ✅ | 0.9663 | 0.8622 | 0.9800 | 0.7696 |
| 7 | MultinomialNB | SMOTE | ✅ | 0.9598 | 0.8614 | 0.8169 | 0.9110 |
| 8 | MultinomialNB | RandomOver | ✅ | 0.9598 | 0.8621 | 0.8140 | 0.9162 |

### Best Models
-   **Accuracy:** `0.9785` → MultinomialNB (TF-IDF, ngram=`(1,2)`, max_features=`2000`, min_df=`5`)
-   **AUC:** `0.9856` → Logistic Regression (TF-IDF, ngram=`(1,1)`, max_features=`1000`, min_df=`1`)
-   **F1 Score:** `0.9148` → MultinomialNB (TF-IDF, ngram=`(1,2)`, max_features=`2000`, min_df=`5`)

---

## 🔍 Observations

-   **Multinomial Naïve Bayes** + **TF-IDF** + **bigrams** gave the best accuracy and F1 score, with perfect precision but slightly lower recall.
-   **Logistic Regression** achieved the highest AUC, showing strong ranking ability.
-   **ComplementNB** gave the best recall but lower precision (more false positives).
-   Balancing methods (**SMOTE**, **Random Oversampling**) improved recall but reduced accuracy compared to unbalanced models.
-   **SVM** performed well overall but was slower to train compared to Naïve Bayes.

---

## 📚 References

-   Scikit-learn: Working with Text Data
-   CountVectorizer Documentation
-   Naïve Bayes Documentation

---

⚡ This project demonstrates how preprocessing choices, vectorization strategies, and algorithm selection critically affect spam detection performance.
