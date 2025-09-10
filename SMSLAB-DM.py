# SMS Spam Classification Lab: Organized Version

## 1. Load and Explore Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

# Load and prepare dataset
df = pd.read_csv('sms-spam-dataset.csv', encoding='ISO-8859-1')
df.columns = ['Text', 'Class']
df['Label'] = df['Class'].map({'ham': 0, 'spam': 1})
print(df['Class'].value_counts())
df.head()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'].values.astype('U'), df['Label'], test_size=0.25, random_state=42
)

## 2. Define Experiment Function
def run_experiment(vectorizer_type='count', tfidf=False, min_df=1, ngram_range=(1,1),
                   max_features=500, model_type='MultinomialNB', balancing='Unbalanced'):
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(min_df=min_df, lowercase=True, max_features=max_features,
                                     ngram_range=ngram_range, stop_words='english')
    else:
        raise ValueError("Only 'count' vectorizer is supported currently")

    X_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)

    if tfidf:
        tf_transformer = TfidfTransformer()
        X_counts = tf_transformer.fit_transform(X_counts)
        X_test_counts = tf_transformer.transform(X_test_counts)

    if balancing == 'SMOTE':
        sm = SMOTE(random_state=42)
        X_counts, y_train_bal = sm.fit_resample(X_counts, y_train)
    elif balancing == 'RandomOver':
        ro = RandomOverSampler(random_state=42)
        X_counts, y_train_bal = ro.fit_resample(X_counts, y_train)
    else:
        y_train_bal = y_train

    if model_type == 'MultinomialNB':
        clf = MultinomialNB()
    elif model_type == 'ComplementNB':
        clf = ComplementNB()
    elif model_type == 'SVM':
        clf = SVC(kernel='linear', probability=True)
    elif model_type == 'LogisticRegression':
        clf = LogisticRegression(solver='liblinear', max_iter=1000)
    else:
        raise ValueError("Model not supported")

    clf.fit(X_counts, y_train_bal)
    y_pred = clf.predict(X_test_counts)
    y_prob = clf.predict_proba(X_test_counts)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test_counts)

    return {
        'Model': model_type,
        'Balancing': balancing,
        'TF-IDF': tfidf,
        'min_df': min_df,
        'ngram': ngram_range,
        'max_features': max_features,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob),
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Features Used': X_counts.shape[1],
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }

## 3. Run and Show Each Experiment

# Experiment 1
exp1 = run_experiment(min_df=1, max_features=500, ngram_range=(1,1), tfidf=True, model_type='MultinomialNB')
print("Experiment 1:")
print(pd.DataFrame([exp1])[['Model', 'Balancing', 'TF-IDF', 'Accuracy', 'F1', 'Precision', 'Recall']])

# Experiment 2
exp2 = run_experiment(min_df=3, max_features=1000, ngram_range=(1,4), tfidf=False, model_type='MultinomialNB')
print("\nExperiment 2:")
print(pd.DataFrame([exp2])[['Model', 'Balancing', 'TF-IDF', 'Accuracy', 'F1', 'Precision', 'Recall']])

# Experiment 3
exp3 = run_experiment(min_df=5, max_features=2000, ngram_range=(1,2), tfidf=True, model_type='MultinomialNB')
print("\nExperiment 3:")
print(pd.DataFrame([exp3])[['Model', 'Balancing', 'TF-IDF', 'Accuracy', 'F1', 'Precision', 'Recall']])

# Experiment 4
exp4 = run_experiment(min_df=1, max_features=1000, ngram_range=(1,1), tfidf=True, model_type='ComplementNB')
print("\nExperiment 4:")
print(pd.DataFrame([exp4])[['Model', 'Balancing', 'TF-IDF', 'Accuracy', 'F1', 'Precision', 'Recall']])

# Experiment 5
exp5 = run_experiment(min_df=1, max_features=1000, ngram_range=(1,1), tfidf=True, model_type='SVM')
print("\nExperiment 5:")
print(pd.DataFrame([exp5])[['Model', 'Balancing', 'TF-IDF', 'Accuracy', 'F1', 'Precision', 'Recall']])

# Experiment 6
exp6 = run_experiment(min_df=1, max_features=1000, ngram_range=(1,1), tfidf=True, model_type='LogisticRegression')
print("\nExperiment 6:")
print(pd.DataFrame([exp6])[['Model', 'Balancing', 'TF-IDF', 'Accuracy', 'F1', 'Precision', 'Recall']])

# Experiment 7
exp7 = run_experiment(min_df=1, max_features=1000, ngram_range=(1,1), tfidf=True,
                      model_type='MultinomialNB', balancing='SMOTE')
print("\nExperiment 7:")
print(pd.DataFrame([exp7])[['Model', 'Balancing', 'TF-IDF', 'Accuracy', 'F1', 'Precision', 'Recall']])

# Experiment 8
exp8 = run_experiment(min_df=1, max_features=1000, ngram_range=(1,1), tfidf=True,
                      model_type='MultinomialNB', balancing='RandomOver')
print("\nExperiment 8:")
print(pd.DataFrame([exp8])[['Model', 'Balancing', 'TF-IDF', 'Accuracy', 'F1', 'Precision', 'Recall']])

## 4. Summary Table
results_df = pd.DataFrame([exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8])
results_df[['Model', 'Balancing', 'TF-IDF', 'min_df', 'ngram', 'max_features',
            'Accuracy', 'AUC', 'F1', 'Precision', 'Recall', 'Features Used']]



# Display confusion matrix helper


# Plot confusion matrices



# Print confusion matrices (text only)
def print_conf_matrix(cm, title):
    print(f"\n{title}")
    print("True\Predicted | Ham | Spam")
    print(f"        Ham    | {cm[0,0]:3} | {cm[0,1]:4}")
    print(f"        Spam   | {cm[1,0]:3} | {cm[1,1]:4}")

print_conf_matrix(exp1['Confusion Matrix'], "Confusion Matrix - Experiment 1")
print_conf_matrix(exp2['Confusion Matrix'], "Confusion Matrix - Experiment 2")
print_conf_matrix(exp3['Confusion Matrix'], "Confusion Matrix - Experiment 3")
print_conf_matrix(exp4['Confusion Matrix'], "Confusion Matrix - Experiment 4")
print_conf_matrix(exp5['Confusion Matrix'], "Confusion Matrix - Experiment 5")
print_conf_matrix(exp6['Confusion Matrix'], "Confusion Matrix - Experiment 6")
print_conf_matrix(exp7['Confusion Matrix'], "Confusion Matrix - Experiment 7")
print_conf_matrix(exp8['Confusion Matrix'], "Confusion Matrix - Experiment 8")



# Final Summary Table
print("\nFinal Summary Table:")
summary_cols = ['Model', 'Balancing', 'TF-IDF', 'min_df', 'ngram', 'max_features',
                'Accuracy', 'AUC', 'F1', 'Precision', 'Recall', 'Features Used']
print(results_df[summary_cols].to_string(index=False))

# Export summary table to CSV and Excel
results_df.to_csv("spam_lab_results.csv", index=False)
results_df.to_excel("spam_lab_results.xlsx", index=False)

print("\n✅ Results exported: 'spam_lab_results.csv' and 'spam_lab_results.xlsx'")
