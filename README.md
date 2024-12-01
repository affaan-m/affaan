# Updated README

# **Personal Projects by Affaan**

Welcome to my project repository! Below you'll find a collection of my personal projects showcasing my skills in AI, data analysis, automation, and more. 

---

## **AI Secretary/Consultant/Manager**

A comprehensive automation tool that leverages OpenAI's GPT-3.5-Turbo model and integrates with Google's Gmail and Calendar APIs. The project acts as your personal secretary, consultant, and manager.

### **Key Features**
- **Inbox Prioritization**: Automatically organizes your inbox based on user-defined priorities.
- **AI-Powered Email Suggestions**: Provides actionable insights and recommendations for managing your workflow.
- **Calendar Integration**: Automatically schedules tasks and events in Google Calendar.
- **Personalization**: Customizable to your email credentials and OpenAI API key.

### **Technologies Used**
- OpenAI GPT-3.5-Turbo
- Google APIs (Gmail, Calendar)
- Python Libraries: `google-auth`, `googleapiclient`, `pickle`

### **How It Works**
1. Authorize Gmail and Google Calendar APIs.
2. Fetch emails, categorize them into priorities, and provide action steps.
3. Automate scheduling based on AI-suggested time slots.
4. Send email summaries and next steps directly to your inbox.

---

## **Bitcoin Price Predictor**

### **Holt’s Linear Exponential Smoothing for Time-Series**

An analytical tool to forecast Bitcoin prices using time-series data from Yahoo Finance over the past 5 years.

### **Highlights**
- Explores multiple predictive models including:
  - Holt’s Linear Exponential Smoothing
  - Neural Networks
  - Random Forest
  - ARIMA
  - Linear Regression
- **Validation**: K-Fold Cross-Validation and Bootstrapping.
- **Visualization**: Actual vs. Predicted values plotted for intuitive understanding.

### **Technologies Used**
- Python Libraries: `pandas`, `numpy`, `statsmodels`, `matplotlib`
- Statistical Models: Exponential Smoothing, Regression Analysis

---

### **Bitcoin Price Predictor Using Neural Networks**

A more advanced approach leveraging TensorFlow to predict Bitcoin prices based on historical data.

### **Key Features**
- Implements neural networks with dense layers for regression analysis.
- Evaluates model performance using RMSE (Root Mean Squared Error).
- Visualizes training and validation losses over epochs.

---

## **Budget Calculator**

A Python-based tool to track personal finances by calculating income and expenses to determine your remaining budget.

### **Features**
- **Manual Inputs**: Add income and expenses directly.
- **CSV Support**: Analyze financial data from uploaded CSV files.
- **Budget Alerts**: Alerts you when expenses exceed your budget.

### **How to Use**
1. Define income and expense sets in the script or upload a CSV.
2. Run the script to calculate remaining budget or overage.
3. Get insights on spending habits.

---

## **Instagram Follower Checker**

This tool identifies Instagram accounts that you follow but do not follow you back.

### **Features**
- **Set Comparison**: Compares your following and followers to identify discrepancies.
- **Export Results**: Outputs a text file listing users who do not follow you back.
- **Automation**: Logs in using `instaloader` and processes data seamlessly.

### **How to Use**
1. Install `instaloader` and `pandas` via pip.
2. Authenticate with your Instagram account.
3. Run the script to generate a `not_following_back.txt` file in your user directory.

---

## **Sentiment and Crypto Price Tracker**

A real-time tracker to display Bitcoin prices and their sentiment on an OLED display.

### **Features**
- **Sentiment Analysis**: Integrates preprocessed sentiment data using Python's VADER.
- **Real-Time Price Fetching**: Pulls Bitcoin prices from the CoinDesk API.
- **Embedded Display**: Uses an Adafruit OLED display for a clean user interface.

### **Technologies**
- C++ (Arduino)
- Python (Data Preprocessing and Sentiment Analysis)
- APIs: CoinDesk, Custom Sentiment Endpoint

---

## **Other Highlights**
### **Personalized Tools**
- Email Categorization Agent
- Advanced Document Retrieval System
- Financial Tracker

### **Machine Learning Pipelines**
- Integration with LangChain for document parsing.
- Use of embeddings (e.g., OpenAI) for advanced search capabilities.

---

## **Connect with Me**

- **GitHub**: [affaan-m](https://github.com/affaan-m)
- **Instagram**: [Affaan Mustafa](https://www.instagram.com/affaanmustafa/)
- **LinkedIn**: [Affaan Mustafa](https://www.linkedin.com/in/affaanmustafa/)
- **Twitter/X**: [Affaan Mustafa](https://www.x.com/affaanmustafa/)
- **Linktree**: [All Links](https://linqapp.com/affaan_mustafa?r=link)

---

### **About My Company**
Explore DCUBE at [dcube.ai](https://dcube.ai) — a startup specializing in **Personalized and Affordable Data Annotation Services**. We offer innovative AI solutions for businesses of all sizes.

---

**License**: This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
