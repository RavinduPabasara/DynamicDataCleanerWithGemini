import pandas as pd
import streamlit as st
import os
import logging
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Set the Google API key
# os.environ['GOOGLE_API_KEY'] = "APIKEY"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# Prompt template
prompt1 = PromptTemplate(
    input_variables=['sample'],
    template=("Here is a sample of my dataset:\n{sample}\n"
              "Please provide a list of cleaning steps that are only essential and suitable to this dataset "
              "using these numbers (return only the numbers, separated by commas):\n"
              "1=remove numbering\n2=remove unwanted spacing\n3=convert to lowercase\n"
              "4=remove special characters\n5=fill missing values with 'N/A'\n"
              "6=drop rows with missing values\n7=replace NaN with median")
)

# Initialize the output parser
output_parser = StrOutputParser()

# Define the chain
chain1 = prompt1 | model | output_parser


def get_cleaning_steps(sample):
    """
    Get the list of cleaning steps from the language model.
    """
    try:
        response = chain1.invoke({"sample": sample})
        logging.info(f"Model response: {response}")
        return response.strip().split(',')
    except Exception as e:
        logging.error(f"Error in getting cleaning steps: {e}")
        return []


# Cleaning functions
def remove_numbering(df):
    logging.info("Removing numbering")
    return df.replace(to_replace=r'^\d+', value='', regex=True)


def remove_unwanted_spacing(df):
    logging.info("Removing unwanted spacing")
    return df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


def convert_to_lowercase(df):
    logging.info("Converting to lowercase")
    return df.applymap(lambda x: x.lower() if isinstance(x, str) else x)


def remove_special_characters(df):
    logging.info("Removing special characters")
    return df.replace(to_replace=r'[^a-zA-Z0-9\s]', value='', regex=True)


def fill_missing_values(df):
    logging.info("Filling missing values with 'N/A'")
    return df.fillna('N/A')


def drop_rows_with_missing_values(df):
    logging.info("Dropping rows with missing values")
    return df.dropna()


def replace_nan_with_median(df):
    logging.info("Replacing NaN with median")
    for column in df.select_dtypes(include=['float64', 'int64']):
        median = df[column].median()
        df[column].fillna(median, inplace=True)
    return df


# Mapping cleaning steps to functions
cleaning_functions = {
    '1': remove_numbering,
    '2': remove_unwanted_spacing,
    '3': convert_to_lowercase,
    '4': remove_special_characters,
    '5': fill_missing_values,
    '6': drop_rows_with_missing_values,
    '7': replace_nan_with_median,
}


def clean_dataset(dataset, sample_size=50):
    """
    Clean the dataset based on the cleaning steps recommended by the LLM.

    Args:
        dataset (DataFrame): The dataset to clean.
        sample_size (int): Number of samples to use for generating cleaning steps.
    """
    # Extract a sample
    sample = dataset.sample(n=sample_size, random_state=42).to_string()
    logging.debug(f"Sample extracted: \n{sample}")

    # Get cleaning steps from LLM
    cleaning_steps = get_cleaning_steps(sample)
    logging.info(f"Cleaning steps: {cleaning_steps}")

    # Apply cleaning steps to the dataset
    for step in cleaning_steps:
        step = step.strip()
        if step in cleaning_functions:
            try:
                dataset = cleaning_functions[step](dataset)
            except Exception as e:
                logging.error(f"Error in applying cleaning step {step}: {e}")

    logging.info("Cleaning process completed")
    return dataset


# Streamlit app
st.title("Dynamic Data Cleaner")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    dataset = pd.read_csv(uploaded_file)
    st.write("Original Dataset", dataset)

    # Sample size input
    sample_size = st.number_input("Sample Size", min_value=1, max_value=len(dataset), value=50)

    # Clean the dataset
    if st.button("Clean Dataset"):
        cleaned_dataset = clean_dataset(dataset, sample_size)
        st.write("Cleaned Dataset", cleaned_dataset)

        # Provide download link for the cleaned dataset
        cleaned_csv = cleaned_dataset.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Cleaned CSV", data=cleaned_csv, file_name='cleaned_dataset.csv',
                           mime='text/csv')
