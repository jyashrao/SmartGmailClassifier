import os
import io
import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

# Google API Imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# --- SETUP & ML LOGIC ---
nltk.download('punkt', quiet=True)
model = joblib.load('spam_model.pkl')
cv = joblib.load('vectorizer.pkl')

reg = RegexpTokenizer(r'[a-z]+')
stemmer = PorterStemmer()

def clean_text_pipeline(text):
    string = str(text).lower()
    tokens = reg.tokenize(string)
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

# --- GMAIL API LOGIC ---
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail(auto_start=False):
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif auto_start:
            # Automatic login trigger
            chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
            if not os.path.exists(chrome_path):
                chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
            
            os.environ['BROWSER'] = chrome_path + ' %s'
            
            if os.path.exists('credentials.json'):
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                st.info("🚀 Redirecting to Google Login... Please sign in to continue.")
                creds = flow.run_local_server(port=0)
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
                st.rerun()
            else:
                st.error("credentials.json not found! Please provide it to enable Gmail access.")
                return None
        else:
            return None
            
    return build('gmail', 'v1', credentials=creds)

def get_user_email(service):
    """Fetches the email address of the authenticated user."""
    try:
        profile = service.users().getProfile(userId='me').execute()
        return profile.get('emailAddress')
    except Exception:
        return "Unknown User"

def fetch_emails(service, max_results=20):
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    email_data = []

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        payload = msg_data.get('payload', {})
        headers = payload.get('headers', [])
        
        subject = "No Subject"
        sender = "Unknown"
        date = "Unknown"
        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            if header['name'] == 'From':
                sender = header['value']
            if header['name'] == 'Date':
                date = header['value']
                
        snippet = msg_data.get('snippet', '')
        email_data.append({'Sender': sender, 'Subject': subject, 'Content': snippet, 'Date': date})
        
    return pd.DataFrame(email_data)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Professional Email Classifier", layout="wide", page_icon="📧")

st.title("📧 Smart Gmail Classifier & Organizer")
st.markdown("""
    **Transform your chaotic inbox into an organized workspace.** 
    This tool uses AI to classify your emails into categories like **Job**, **Finance**, and **Spam** instantly.
            
            Minor Project by Yash & Jeet
""")

# --- AUTHENTICATION FLOW ---
service = None
if not os.path.exists('token.json'):
    st.warning("🔒 You are not logged in to Gmail.")
    if st.button("Connect your Gmail Account Now"):
        service = authenticate_gmail(auto_start=True)
else:
    try:
        service = authenticate_gmail()
        current_user = get_user_email(service)
        st.sidebar.success(f"✅ **Connected:** {current_user}")
        if st.sidebar.button("Logout / Switch Account"):
            os.remove('token.json')
            st.rerun()
    except Exception:
        st.sidebar.error("Session expired.")
        if st.sidebar.button("Reconnect"):
            os.remove('token.json')
            st.rerun()

st.divider()

# --- MAIN CONTROLS ---
col1, col2 = st.columns([2, 1])

with col1:
    num_emails = st.slider("Select number of recent emails to analyze:", 5, 100, 20)

with col2:
    st.write("") # Spacing
    fetch_btn = st.button("🚀 Fetch and Classify Emails", use_container_width=True)

if fetch_btn:
    if not service:
        st.error("Please login first to fetch emails.")
    else:
        with st.spinner("🔍 Accessing Gmail and running AI Classification..."):
            try:
                # 1. Fetch Emails
                df = fetch_emails(service, max_results=num_emails)
                
                # 2. Clean Text and Predict
                cleaned_content = df['Content'].apply(clean_text_pipeline)
                vectorized_content = cv.transform(cleaned_content).toarray()
                predictions = model.predict(vectorized_content)
                
                # 3. Labeling Logic
                def get_final_label(row, pred):
                    if pred == 1:
                        return "🚫 IGNORE (SPAM)"
                    
                    text_to_check = (str(row['Subject']) + " " + str(row['Content'])).lower()
                    
                    # IMPORTANT
                    if any(word in text_to_check for word in ['urgent', 'important', 'action required', 'meeting', 'deadline', 'priority']):
                        return "⭐ IMPORTANT"
                    # JOB
                    elif any(word in text_to_check for word in ['job', 'hiring', 'career', 'interview', 'vacancy', 'offer', 'internship', 'application']):
                        return "💼 JOB"
                    # FINANCE
                    elif any(word in text_to_check for word in ['bank', 'otp', 'transaction', 'payment', 'invoice', 'statement', 'bill', 'credit', 'debit']):
                        return "💳 FINANCE"
                    else:
                        return "✅ GENERAL / SAFE"

                df['Classification'] = [get_final_label(row, p) for row, p in zip(df.to_dict('records'), predictions)]
                
                # Reorder columns for display
                display_df = df[['Classification', 'Sender', 'Subject', 'Date', 'Content']]
                
                st.success(f"Analyzed {len(df)} emails successfully!")
                st.dataframe(display_df, use_container_width=True)
                
                # 4. Excel Download
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Classified Emails')
                
                st.download_button(
                    label="📥 Download Classified Data as Excel",
                    data=buffer,
                    file_name="Classified_Emails_Report.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Tip: Make sure your 'credentials.json' is valid.")