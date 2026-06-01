import os
import io
import json
import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

# Google API Imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# --- SETUP & ML LOGIC ---
@st.cache_resource
def load_resources():
    nltk.download('punkt', quiet=True)
    model = joblib.load('spam_model.pkl')
    cv = joblib.load('vectorizer.pkl')
    return model, cv

model, cv = load_resources()

reg = RegexpTokenizer(r'[a-z]+')
stemmer = PorterStemmer()

def clean_text_pipeline(text):
    string = str(text).lower()
    tokens = reg.tokenize(string)
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

# --- GMAIL API LOGIC ---
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
# Replace this with your actual app URL if it changes
REDIRECT_URI = "https://smartgmailclassifier-jyashrao.streamlit.app"

def get_gmail_service():
    """Returns a Gmail service object based on session credentials."""
    if "credentials" not in st.session_state:
        return None
    
    creds = st.session_state["credentials"]
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            st.session_state["credentials"] = creds
        except Exception:
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
# 1. Handle OAuth Redirect Callback
if "code" in st.query_params and "credentials" not in st.session_state:
    if "GOOGLE_CREDENTIALS" in st.secrets:
        try:
            creds_info = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
            flow = Flow.from_client_config(creds_info, scopes=SCOPES, redirect_uri=REDIRECT_URI)
            flow.fetch_token(code=st.query_params["code"])
            st.session_state["credentials"] = flow.credentials
            # Clear the code from URL for a clean refresh
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Failed to fetch token: {e}")

# 2. Check for active session
service = get_gmail_service()

if not service:
    st.sidebar.warning("🔒 Account not connected.")
    
    # Try to use Global Secret if it exists and session is empty (Demo Mode)
    if "GOOGLE_TOKEN" in st.secrets and "credentials" not in st.session_state:
        st.sidebar.info("Using Demo Account from Secrets.")
        token_info = json.loads(st.secrets["GOOGLE_TOKEN"])
        st.session_state["credentials"] = Credentials.from_authorized_user_info(token_info, SCOPES)
        st.rerun()

    if "GOOGLE_CREDENTIALS" in st.secrets:
        try:
            creds_info = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
            flow = Flow.from_client_config(creds_info, scopes=SCOPES, redirect_uri=REDIRECT_URI)
            auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
            
            st.markdown(f"### 🛡️ Secure Login Required")
            st.write("To analyze **your** emails, please connect your Gmail account securely using the button below.")
            st.link_button("🚀 Connect your Gmail Account", auth_url, type="primary", use_container_width=True)
            st.info("💡 Note: We only have 'Read-Only' access to your emails. We cannot send or delete anything.")
        except Exception as e:
            st.error(f"Setup Error: {e}")
    else:
        st.error("Missing GOOGLE_CREDENTIALS in Secrets.")
else:
    current_user = get_user_email(service)
    st.sidebar.success(f"✅ **Connected:** {current_user}")
    if st.sidebar.button("Logout / Switch Account"):
        # Clear per-session credentials
        del st.session_state["credentials"]
        # Tip: Tell user to remove GOOGLE_TOKEN from secrets if they want true privacy
        st.sidebar.info("Logged out from this session.")
        st.rerun()

st.divider()

# --- MAIN CONTROLS ---
if service:
    col1, col2 = st.columns([2, 1])

    with col1:
        num_emails = st.slider("Select number of recent emails to analyze:", 5, 100, 20)

    with col2:
        st.write("") # Spacing
        fetch_btn = st.button("🚀 Fetch and Classify Emails", use_container_width=True)

    if fetch_btn:
        with st.spinner("🔍 Accessing Gmail and running AI Classification..."):
            try:
                # 1. Fetch Emails
                df = fetch_emails(service, max_results=num_emails)
                
                if df.empty:
                    st.warning("No emails found.")
                else:
                    # 2. Clean Text and Predict
                    cleaned_content = df['Content'].apply(clean_text_pipeline)
                    vectorized_content = cv.transform(cleaned_content).toarray()
                    predictions = model.predict(vectorized_content)
                    
                    # 3. Labeling Logic
                    def get_final_label(row, pred):
                        if pred == 1:
                            return "🚫 IGNORE (SPAM)"
                        
                        text_to_check = (str(row['Subject']) + " " + str(row['Content'])).lower()
                        
                        if any(word in text_to_check for word in ['urgent', 'important', 'action required', 'meeting', 'deadline', 'priority']):
                            return "⭐ IMPORTANT"
                        elif any(word in text_to_check for word in ['job', 'hiring', 'career', 'interview', 'vacancy', 'offer', 'internship', 'application']):
                            return "💼 JOB"
                        elif any(word in text_to_check for word in ['bank', 'otp', 'transaction', 'payment', 'invoice', 'statement', 'bill', 'credit', 'debit']):
                            return "💳 FINANCE"
                        else:
                            return "✅ GENERAL / SAFE"

                    df['Classification'] = [get_final_label(row, p) for row, p in zip(df.to_dict('records'), predictions)]
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
else:
    st.info("Please login to see the analysis controls.")
