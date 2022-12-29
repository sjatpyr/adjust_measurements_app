import streamlit 
import pandas 
import xgboost
import utils

from pathlib import Path

# Streamlit configs
streamlit.set_page_config(layout="wide")

# Environment variables
PREDICTION_MODEL_PATH = "trained_model.json"
EXAMPLE_TEST_DATA_PATH = "test_data_example.csv"
TEST_DATA_PATH = "test_data.csv"
OUTPUT_FILE_NAME = "file.csv"

if __name__ == "__main__":
    # Inputs 
    ## Load example and test data
    example_df = pandas.read_csv(EXAMPLE_TEST_DATA_PATH)
    ## Load test data
    uploaded_file = streamlit.sidebar.file_uploader("**Upload your input CSV file**", type=["csv"])
    if uploaded_file is not None:
        input_df = pandas.read_csv(uploaded_file)
    else:
        input_df = example_df.copy()
    ## Load prediction model
    model = xgboost.XGBRegressor()
    model.load_model(PREDICTION_MODEL_PATH)

    # App
    streamlit.write("""
        # Jūsu sugalvotas pavadinimas <...>
        ### Description
        This app converts the measurement values obtained by both spectrometers into the reference values by using the prediction model.
        ### How to use it?
        Įkelti CSV failą su duotomis dirbinio sudėties reikšmėmis (%): įkelti CSV failą paspaudžiant "Browse file".
    """)

    ## Displays the user input features
    streamlit.write("""
        ### How Input Data Should Look Like?
        It's the required input format for the Input CSV file. You can provide one or more input rows. See the example below.
        - 0's and 1's in is_Niton and is_Solid refer to "No" and "Yes", respectively.
    """)
    streamlit.write(example_df)

    ## Read input data and use model to make predictions
    ### Collects user input features into dataframe
    streamlit.subheader("Given User Input Values")
    streamlit.write("""
        - If CSV file is not provided, input data and predictions are generated using the example dataset.
    """)
    streamlit.write(input_df)

    ### Predictions
    predictions = utils.predict(model, input_df)
    streamlit.subheader("Adjusted Input Values")
    streamlit.write(predictions)

    ### Allow to download prediction data
    @streamlit.experimental_memo
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(predictions)
    streamlit.download_button(
        "Press to Download Adjusted Values",
        csv,
        OUTPUT_FILE_NAME,
        "text/csv",
        key='download-csv'
    )
