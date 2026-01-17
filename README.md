<p align="center">
  <img src="logo.jpg" alt="PhishGuard Logo" width="200">
</p>

<h1 align="center"> PhishGuard </h1>

<p align="center">
  <strong>A professional, high-performance AI system for detecting phishing attempts with precision.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Phishing%20Detection-Advanced-red" alt="Phishing Detection">
  <img src="https://img.shields.io/badge/Dataset-18k%20Emails-blue" alt="Dataset">
  <img src="https://img.shields.io/badge/UI-Modern%20Streamlit-green" alt="UI">
</p>

---

## ⚙️ About the Project 

**AI Phishing Guard** is a modernized, robust phishing detection system. It leverages **Multinomial Naive Bayes** to analyze email content and suspicious URL patterns, providing instant verification of potential threats. 

The system has been upgraded to support large-scale datasets (18,000+ emails) while maintaining a lightning-fast user experience through intelligent sampling and text processing optimizations.

## 🚀 Key Features

- **Modernized Interface**: A clean, professional UI with a dedicated sidebar for tools and instructions.
- **Large-Scale Intelligence**: Pre-trained on a comprehensive dataset of over 18,000 safe and phishing emails.
- **Instant Examples**: One-click "Try Examples" buttons (Safe vs. Phishing) for immediate testing.
- **Dual-Channel Analysis**: 
    - **Text Mining**: Analyzes tone, urgency, and keywords.
    - **URL Inspection**: Extracts and evaluates links for suspicious protocols and structures.
- **Performance Optimized**: Features intelligent text truncation and sampling to ensure high-speed processing without compromising accuracy.

## 🧑‍💻 Technologies Used

- **Python 3.12+** 🐍
- **Streamlit** (Modern Web UI) 🌐
- **Scikit-learn** (Machine Learning) 🤖
- **Pandas** (Data Engineering) 📊
- **NLTK** (Natural Language Processing) 🧠
- **Joblib** (Model Serialization) 💾

## 🎯 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Phishing_Emails_Detection_Using_Machine_Learning.git
   cd Phishing_Emails_Detection_Using_Machine_Learning
   ```

2. **Setup Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Launch the Guard**
   ```bash
   streamlit run phishing_detection.py
   ```

4. **Run Model Evaluation (Optional)**
   To see detailed metrics (Accuracy, Precision, Recall, F1), run:
   ```bash
   python evaluate_metrics.py
   ```

## 🔍 How It Works

1.  **Automated Training**: On first launch, the system automatically processes the `Phishing_Email.csv` dataset, optimizing it for speed and accuracy.
2.  **Analysis**: When an email is pasted, the system extracts text features and URL patterns.
3.  **Visualization**: Results are displayed in color-coded cards (Green for Safe, Red for Warning) with clear explanations of the AI's logic.

## 📊 Performance & Accuracy

- **Dataset Size**: Supporting over 18,000+ labeled examples.
- **Inference Speed**: Results in < 500ms.
- **Optimizations**: Uses status-check bypassing during bulk training and text truncation (5,000 chars) for stable performance on legacy hardware.

## 🛠️ Contributing

Contributions are welcome! If you'd like to improve the detection logic or UI, please submit a pull request.

## 🙌 Acknowledgements

- **Streamlit** for the powerful UI framework.
- **Scikit-learn** for the ML pipeline.
- **NLTK** for language processing tools.