import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load the pre-trained model
def load_model():
    try:
        model = joblib.load('nlp_model_pipeline.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Streamlit app UI
st.title('NLP Text Classification App')
st.write('This app predicts the target based on text data using a pre-trained NLP model.')

# Load the model
model = load_model()

if model:
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file with the text data.", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load the uploaded file
            test = pd.read_csv(uploaded_file)

            # Check for required columns
            if 'text' not in test.columns:
                st.error("The uploaded CSV file must contain a 'text' column.")
            else:
                # Fill missing values
                test['text'].fillna('', inplace=True)

                # Make predictions
                test_predictions = model.predict(test['text'])

                # Prepare results
                results = pd.DataFrame({
                    'Id': test.index,  # Assuming 'Id' column exists or use index
                    'Prediction': test_predictions
                })

                # Display results
                st.write('Predictions:')
                st.dataframe(results)

                # Option to download the predictions as a CSV file
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions as CSV", csv, "text_classification_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing the file: {e}")

    st.write("Make sure the CSV file contains a 'text' column for predictions.")
else:
    st.error("Model could not be loaded. Please check the file path and compatibility.")
