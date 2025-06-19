import streamlit as st
import pandas as pd
import io

from utils.sentiment_analysis import analyze_sentiment_batch
from utils.visualizations import plot_sentiment_distribution
from utils.export import export_to_csv, export_to_json, export_to_pdf

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("Sentiment Analysis Dashboard")
st.write("""
Analyze the emotional tone of your text data like customer reviews, social media posts, or any text content.
""")

# Sidebar for inputs and export
st.sidebar.header("Input Options")

input_method = st.sidebar.radio("Choose input method:", ("Manual Text Input", "Upload Text File (CSV/TXT)"))

texts = []

if input_method == "Manual Text Input":
    user_input = st.text_area("Enter one or multiple sentences (one per line):", height=150)
    if user_input.strip():
        texts = [line.strip() for line in user_input.split("\n") if line.strip()]

else:
    uploaded_file = st.sidebar.file_uploader("Upload your text file (CSV with 'text' column or TXT):", type=['csv', 'txt'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.sidebar.error("CSV file must have a 'text' column.")
                else:
                    texts = df['text'].dropna().astype(str).tolist()
            else:  # TXT file
                txt = uploaded_file.read().decode('utf-8')
                texts = [line.strip() for line in txt.split("\n") if line.strip()]
        except Exception as e:
            st.sidebar.error(f"Failed to read file: {e}")

if texts:
    with st.spinner("Analyzing sentiment..."):
        results = analyze_sentiment_batch(texts)

    # Convert results to DataFrame for display/export
    df_results = pd.DataFrame(results)

    st.subheader("Analysis Results")
    st.dataframe(df_results[["text", "sentiment", "positive_score", "negative_score", "confidence"]])

    st.subheader("Sentiment Distribution")
    img_str = plot_sentiment_distribution(results)
    st.image(f"data:image/png;base64,{img_str}")

    st.sidebar.header("Export Results")
    col1, col2, col3 = st.sidebar.columns(3)

    if col1.button("Export CSV"):
        export_to_csv(results)
        st.sidebar.success("Exported results to sentiment_results.csv")

    if col2.button("Export JSON"):
        export_to_json(results)
        st.sidebar.success("Exported results to sentiment_results.json")

    if col3.button("Export PDF"):
        export_to_pdf(results)  # Save PDF
        with open("sentiment_results.pdf", "rb") as f:
            st.sidebar.download_button(
                label="Download PDF",
                data=f,
                file_name="sentiment_results.pdf",
                mime="application/pdf"
            )

else:
    st.info("Please input or upload text data to analyze.")
