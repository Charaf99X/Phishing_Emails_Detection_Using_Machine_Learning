import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from nltk.corpus import stopwords
import nltk
import os

# Download NLTK data
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words("english"))

def clean_email_body(email_body):
    # Ensure email_body is a string
    email_body = str(email_body) if email_body is not None else ""
    # Removing non-alphabetic characters
    email_body = re.sub(r'[^a-zA-Z\s]', '', email_body)
    # Lowercasing and removing stopwords
    email_body = ' '.join([word.lower() for word in email_body.split() if word.lower() not in STOPWORDS])
    return email_body

def extract_url_features(email_body):
    # Simplified version for training performance
    email_body = str(email_body) if email_body is not None else ""
    urls = re.findall(r'(https?://[^\s]+)', email_body)
    features = []
    for url in urls:
        features.append(len(url))  # Simplified numeric feature
    return np.mean(features) if features else 0

def run_evaluation():
    if not os.path.exists('Phishing_Email.csv'):
        print("Error: Phishing_Email.csv not found!")
        return

    print("Loading and Preprocessing Dataset...")
    data = pd.read_csv('Phishing_Email.csv')
    
    # Handle column names
    if 'Email Text' in data.columns:
        data = data.rename(columns={'Email Text': 'email_body'})
    if 'Email Type' in data.columns:
        data = data.rename(columns={'Email Type': 'label'})
    
    # Drop nulls
    data = data.dropna(subset=['email_body'])
    
    # Map labels to numeric
    if data['label'].dtype == 'object':
        data['label'] = data['label'].apply(lambda x: 1 if 'Phishing' in str(x) else 0)

    # Use a sample for speed (20k as in original optimization)
    if len(data) > 20000:
        data = data.sample(20000, random_state=42)

    # Truncate emails for consistency
    data['email_body'] = data['email_body'].astype(str).str[:5000]
    
    print("Extracting features...")
    data['cleaned_body'] = data['email_body'].apply(clean_email_body)
    data['url_features'] = data['email_body'].apply(extract_url_features)

    X = data['cleaned_body']
    y = data['label']

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Model for Evaluation...")
    vectorizer = CountVectorizer()
    clf = MultinomialNB()
    model = make_pipeline(vectorizer, clf)
    model.fit(X_train, y_train)

    print("Generating Predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "="*40)
    print("📈 MODEL EVALUATION RESULTS")
    print("="*40)
    
    results = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]
    }
    
    df_metrics = pd.DataFrame(results)
    print(df_metrics.to_string(index=False))
    print("="*40)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))
    
    return df_metrics

if __name__ == "__main__":
    run_evaluation()
