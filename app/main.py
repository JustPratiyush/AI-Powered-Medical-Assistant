import streamlit as st
import sys
import os

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag_pipeline import get_medical_response

# Page configuration
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark blue theme
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #0a1929;
    }
    .diagnosis-box {
        background-color: #112940;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-top: 20px;
        color: #e0e0e0;
        border-left: 4px solid #4d94ff;
    }
    .sidebar-content {
        padding: 20px;
        color: #e0e0e0;
    }
    .title-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        color: #ffffff;
    }
    .footer-text {
        font-size: 14px;
        color: #b0b0b0;
        text-align: center;
        margin-top: 30px;
    }
    .developer-info {
        background-color: #15375e;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        text-align: center;
        border-left: 4px solid #4d94ff;
        color: #ffffff;
    }
    /* Override Streamlit elements for dark theme */
    .stTextInput, .stTextArea, .stSelectbox {
        background-color: #112940;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4d94ff;
        color: #ffffff;
    }
    .stExpander {
        background-color: #112940;
        color: #e0e0e0;
    }
    .stMarkdown {
        color: #e0e0e0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    st.image("./assets/dog.jpeg", width=200)
    st.markdown("""
    **Developed by:**  
    Abhinav Kuchhal  
    CCE Department, 4th Semester  
    Registration No: 23FE10CCE00063
    """)
    st.markdown("## About")
    st.markdown("""
    This AI Medical Assistant uses advanced LLM technology 
    combined with RAG (Retrieval Augmented Generation) to provide 
    preliminary medical insights based on reported symptoms.
    
    **Disclaimer:** This tool is for informational purposes only 
    and is not a substitute for professional medical advice, 
    diagnosis, or treatment.
    """)
    
    st.markdown("### How to use")
    st.markdown("""
    1. Enter your symptoms in the text area
    2. Be as detailed as possible
    3. Click "Analyze Symptoms"
    4. Review the AI-generated information
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Main content
st.markdown("""
    <div class="title-container">
        <h1>🩺 AI-Powered Medical Assistant</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    This tool helps you understand possible medical conditions based on your symptoms.
    Always consult with a healthcare professional for proper diagnosis and treatment.
""")

# User input section
st.markdown("### Describe Your Symptoms")
user_input = st.text_area(
    "Please be detailed about what you're experiencing (location, duration, severity, etc.)",
    height=150,
    placeholder="Example: I've been experiencing a persistent headache on the right side of my head for 3 days, along with slight nausea in the mornings..."
)

# Additional context (optional)
with st.expander("Additional Information (Optional)"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
    with col2:
        medical_history = st.multiselect(
            "Existing Medical Conditions",
            ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "Cancer", "None"]
        )
        medications = st.text_input("Current Medications (if any)")

# Analysis button with loading state
analyze_button = st.button("Analyze Symptoms", type="primary", use_container_width=True)

if analyze_button:
    if user_input.strip():  # Ensures input is not just whitespace
        with st.spinner("Analyzing your symptoms..."):
            # Prepare additional context if provided
            context = ""
            if age > 0:
                context += f"Age: {age}. "
            if gender:
                context += f"Gender: {gender}. "
            if medical_history and "None" not in medical_history:
                context += f"Medical history: {', '.join(medical_history)}. "
            if medications:
                context += f"Current medications: {medications}."
                
            # Combine symptom description with context if available
            query = user_input
            if context:
                query += f"\n\nAdditional context: {context}"
                
            # Get response from RAG pipeline
            response = get_medical_response(query)
            
            # Display results
            st.markdown("<div class='diagnosis-box'>", unsafe_allow_html=True)
            st.subheader("📋 Analysis Results")
            st.markdown(response)
            
            st.markdown("""
            ---
            **Important:** This is an AI-generated analysis and should not replace 
            professional medical advice. If you're experiencing severe symptoms, 
            please seek immediate medical attention.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter your symptoms before analyzing.")

# Footer
st.markdown("---")
st.markdown(
    "<div class='footer-text'>Powered by LLaMA + RAG technology. For emergencies, call your local emergency number immediately.</div>",
    unsafe_allow_html=True
)

# Developer attribution at the bottom of the main page as well
st.markdown("""
<div class='developer-info'>
    <strong>Developed by:</strong> Abhinav Kuchhal | CCE Department, 4th Semester | Registration No: 23FE10CCE00063
</div>
""", unsafe_allow_html=True)