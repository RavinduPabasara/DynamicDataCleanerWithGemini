
# Dynamic Data Cleaner

## Overview

The Dynamic Data Cleaner is a web application built using Streamlit that leverages LangChain and Google Generative AI to test the ability of a LLM to automatically suggest and apply data cleaning steps to a CSV dataset. The application provides an intuitive interface to upload a dataset, receive recommendations on essential cleaning steps, and download the cleaned dataset.

## Features

- **Upload CSV File:** Allows users to upload their dataset in CSV format.
- **Automatic Cleaning Recommendations:** Utilizes a Generative AI model to analyze a sample of the dataset and suggest appropriate cleaning steps.
- **Apply Cleaning Steps:** Applies the recommended cleaning steps to the dataset.
- **Download Cleaned Dataset:** Provides a download link for the cleaned CSV file.

## Prerequisites

Before running the application, ensure you have the following:

- Python 3.8 or higher
- Google API Key (for using Google Generative AI)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/RavinduPabasara/DynamicDataCleanerWithGemini
   cd <repository-directory>
   ```

2. **Install Dependencies**

   It is recommended to create a virtual environment before installing dependencies.

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Set Up Google API Key**

   Replace the placeholder Google API key in `app.py` and `main.py` with your actual API key:

   ```python
   os.environ['GOOGLE_API_KEY'] = "your_google_api_key"
   ```

## Usage

1. **Run the Application**

   Start the Streamlit app by running:

   ```bash
   streamlit run app.py
   ```

2. **Interact with the App**

   - **Upload a CSV file:** Click on the "Upload your CSV file" button and select your file.
   - **Set Sample Size:** Specify the number of samples to use for generating cleaning steps.
   - **Clean Dataset:** Click the "Clean Dataset" button to process your file based on the AI-generated cleaning steps.
   - **Download Cleaned CSV:** After processing, download the cleaned dataset using the provided download button.

## Cleaning Steps

The application can suggest the following cleaning steps:

1. **Remove Numbering:** Remove numbering from the dataset.
2. **Remove Unwanted Spacing:** Trim extra spaces from the data.
3. **Convert to Lowercase:** Convert all text to lowercase.
4. **Remove Special Characters:** Remove any non-alphanumeric characters.
5. **Fill Missing Values with 'N/A':** Replace missing values with 'N/A'.
6. **Drop Rows with Missing Values:** Remove rows that have missing values.
7. **Replace NaN with Median:** Replace NaN values with the median of the respective column.

## Troubleshooting

- **Error in Getting Cleaning Steps:** Check if the Google API key is set correctly and that your API quota is not exceeded.
- **Error in Applying Cleaning Steps:** Ensure that the dataset format is compatible and does not contain unsupported data types.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit issues, feature requests, or pull requests. Contributions are welcome!

## Contact

For any questions or feedback, please contact [karurpabe@gmail.com].

```
