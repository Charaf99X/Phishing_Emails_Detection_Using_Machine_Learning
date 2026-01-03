import pandas as pd
import numpy as np
import re
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from nltk.corpus import stopwords
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

# Download NLTK data
import nltk
nltk.download('stopwords')

# Load stopwords
STOPWORDS = set(stopwords.words("english"))

# Function to clean and process the email body
def clean_email_body(email_body):
    # Ensure email_body is a string
    email_body = str(email_body) if email_body is not None else ""
    # Removing non-alphabetic characters
    email_body = re.sub(r'[^a-zA-Z\s]', '', email_body)
    # Lowercasing and removing stopwords
    email_body = ' '.join([word.lower() for word in email_body.split() if word.lower() not in STOPWORDS])
    return email_body

def extract_url_features(email_body, check_status=True):
    # Ensure email_body is a string
    email_body = str(email_body) if email_body is not None else ""
    urls = re.findall(r'(https?://[^\s]+)', email_body)
    features = []
    
    for url in urls:
        parsed_url = urlparse(url)
        
        # Ensure numeric values are appended to the features list
        domain_length = len(parsed_url.netloc)  # Length of the domain
        path_length = len(parsed_url.path)  # Length of the path
        protocol = parsed_url.scheme  # Protocol (http/https)
        
        features.append(domain_length)
        features.append(path_length)
        
        # Check if the protocol is valid and append to features
        if protocol in ['http', 'https']:
            features.append(1)
        else:
            features.append(0)
        
        # Skip status check during training or if disabled
        if check_status:
            try:
                response = requests.get(url, timeout=3)
                status_code = response.status_code
                features.append(status_code)
            except:
                features.append(0)  # If URL is not reachable, append 0
    
    # Return the mean of numeric values in the features list or 0 if empty
    return np.mean([f for f in features if isinstance(f, (int, float))]) if features else 0

# Function to load and train the model
def train_model(data):
    # Handle different column naming schemes
    if 'Email Text' in data.columns:
        data = data.rename(columns={'Email Text': 'email_body'})
    if 'Email Type' in data.columns:
        data = data.rename(columns={'Email Type': 'label'})
    
    # Drop rows with missing email body
    data = data.dropna(subset=['email_body'])
    
    # Convert text labels to numeric if needed
    if data['label'].dtype == 'object':
        data['label'] = data['label'].apply(lambda x: 1 if 'Phishing' in str(x) else 0)
    
    # Sample data for performance (limit to 20k rows)
    if len(data) > 20000:
        data = data.sample(20000, random_state=42)
    
    # Truncate emails to 5000 characters for processing speed
    data['email_body'] = data['email_body'].astype(str).str[:5000]
    
    # Clean and process email bodies
    data['cleaned_body'] = data['email_body'].apply(clean_email_body)
    # Disable status checks during bulk training for speed
    data['url_features'] = data['email_body'].apply(lambda x: extract_url_features(x, check_status=False))

    X = pd.concat([data['cleaned_body'], data['url_features']], axis=1)
    X.columns = ['email_body', 'url_features']
    y = data['label']

    # Use CountVectorizer for email body and concatenate URL features
    body_vectorizer = CountVectorizer()
    model = make_pipeline(body_vectorizer, MultinomialNB())

    # Train the model
    model.fit(X['email_body'], y)
    
    # Save the trained model
    joblib.dump(model, 'phishing_model.pkl')
    
    return model

# Function to predict phishing
def predict_phishing(model, email_body):
    # Clean and process the email body and extract URL features
    cleaned_body = clean_email_body(email_body)
    url_features = extract_url_features(email_body)
    X = pd.DataFrame([[cleaned_body, url_features]], columns=['email_body', 'url_features'])
    
    # Predict using the trained model
    prediction = model.predict(X['email_body'])
    return 'Phishing' if prediction[0] == 1 else 'Safe'

# Streamlit interface
def main():
    # Set page configuration
    st.set_page_config(
        page_title="PhishGuard",
        page_icon="🚨",
        layout="centered"
    )

    # Custom CSS for modern look
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextArea textarea {
            border-radius: 10px;
            border: 1px solid #ced4da;
        }
        .prediction-card {
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            margin-top: 20px;
        }
        .safe-card {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .phishing-card {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("logo.jpg", width=150)
        st.title("PhishGuard")
        st.info("""
        **How to use:**
        1. Paste an email body in the text area.
        2. Click 'Analyze for Phishing'.
        3. See the AI prediction instantly.
        """)
        
        st.divider()
        st.subheader("💡 Try Examples")
        
        phishing_example = "URGENT: Your account has been compromised. Please reset your password immediately at http://security-update-validate.com to prevent suspension."
        safe_example = "Hi team, the project deadline has been extended by two days. Please check the updated timeline on the shared drive. Regards, Project Manager."

        if st.button("📥 Load Phishing Example"):
            st.session_state.email_input = phishing_example
            
        if st.button("📥 Load Safe Example"):
            st.session_state.email_input = safe_example

        st.divider()
        # Upload email dataset for training the model
        st.subheader("⚙️ Model Training")
        uploaded_file = st.file_uploader("Update model with CSV", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            # Check for either column naming scheme
            has_old_schema = 'email_body' in data.columns and 'label' in data.columns
            has_new_schema = 'Email Text' in data.columns and 'Email Type' in data.columns
            
            if has_old_schema or has_new_schema:
                with st.spinner('🔄 Training in progress...'):
                    train_model(data)
                st.success("✅ Model updated!")
            else:
                st.error("❌ Invalid CSV format.")

    # Main Content
    st.title("Phishing Detection System")
    st.markdown("---")
    
    # Load trained model if available
    model = None
    try:
        model = joblib.load('phishing_model.pkl')
    except:
        st.warning("⚠️ No trained model found. Please train the model in the sidebar.")
        # Auto-train logic if Phishing_Email.csv exists
        if os.path.exists('Phishing_Email.csv'):
            with st.status("📂 Auto-training on legacy data...", expanded=True) as status:
                st.write("Processing dataset...")
                data = pd.read_csv('Phishing_Email.csv')
                model = train_model(data)
                status.update(label="✅ Model ready!", state="complete", expanded=False)

    # Email Input with session state
    if 'email_input' not in st.session_state:
        st.session_state.email_input = ""

    email_input = st.text_area(
        "Paste the email content below:", 
        value=st.session_state.email_input, 
        height=250,
        placeholder="e.g., Dear customer, your invoice is due..."
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("Analyze for Phishing", use_container_width=True, type="primary")

    if analyze_btn:
        if email_input:
            with st.spinner('Analyzing content...'):
                result = predict_phishing(model, email_input)
            
            if result == 'Phishing':
                st.markdown(f"""
                    <div class="prediction-card phishing-card" style=" color: red;">
                        This appears to be a Phishing Email!
                    </div>
                """, unsafe_allow_html=True)
                st.error("Highly suspicious elements detected: Urgency, suspicious links, or unusual patterns.")
            else:
                st.markdown(f"""
                    <div class="prediction-card safe-card" style=" color: green;">
                        This looks like a Safe Email.
                    </div>
                """, unsafe_allow_html=True)
                st.success("No standard phishing markers identified.")

            with st.expander("ℹWhy this result?"):
                st.write("""
                Our AI model analyzes:
                - **Keywords**: Looks for words typical of phishing (e.g., 'urgent', 'winner', 'account verify').
                - **URL Patterns**: Scans for suspicious characters and protocols in links.
                - **Contextual Clues**: Evaluates the overall structure of the text.
                """)
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == '__main__':
    main()