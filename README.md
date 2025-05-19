 Amazon & Flipkart Price Tracker with Gemini AI Insights

This is a Python-based desktop application built using Tkinter that scrapes product data from Amazon and Flipkart, calculates scores based on price, rating, and reviews, visualizes price history, performs Gemini AI-based product comparison, and sends detailed reports via email.

 Features

- Scrapes product data from Amazon and Flipkart using Selenium
- Calculates a normalized score for each product
- Filters products with score > 6
- Plots price history using Matplotlib
- Performs semantic analysis to ensure product relevance
- Uses Gemini AI to provide insights and comparisons
- Sends product reports with CSV, analysis, and graphs via email

 Prerequisites

- Python 3.8 or higher
- Chrome browser
- ChromeDriver (matching your Chrome version)

Installation

1. **Clone the repository or download the files**.

2. **Install required Python packages**:

   Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate     # On Windows: .venv\Scripts\activate

PATH FOR CHROMEDRIVER
service = Service(r"C:\\path\\to\\chromedriver.exe")
FOR GEMINI AI ANALYSIS
genai.configure(api_key="YOUR_API_KEY")
EMAIL AUTOMATION
sender_email = "your-email@gmail.com"
sender_password = "your-app-password"
FOR EXECUTION SIMPLY EXECUTE :-
python FINALIRPRO.py
requirement.txt
beautifulsoup4==4.12.3
fake-useragent==1.5.1
google-generativeai==0.4.1
matplotlib==3.8.4
pandas==2.2.2
Pillow==10.3.0
selenium==4.21.0
sentence-transformers==2.2.2
torch>=1.10
tk
