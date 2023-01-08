import streamlit 
import pandas 
import numpy
import xgboost
import utils

# Streamlit configs
streamlit.set_page_config(layout="wide")

# Environment variables
PREDICTION_MODEL_PATH = "trained_model.json"
EXAMPLE_TEST_DATA_PATH = "test_data_example.csv"
OUTPUT_FILE_NAME = "file.csv"

if __name__ == "__main__":
    # Inputs 
    ## Load example and test data
    example_df = pandas.read_csv(EXAMPLE_TEST_DATA_PATH)
    ## Load test data
    uploaded_file = streamlit.sidebar.file_uploader("**Įkelk čia įvesties duomenis**", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        input_df = pandas.read_excel(uploaded_file)
    else:
        input_df = example_df.copy()
    ## Load prediction model
    model = xgboost.XGBRegressor()
    model.load_model(PREDICTION_MODEL_PATH)

    # App
    streamlit.write("""
        # RENTGENO FLUORESCENCIJOS SPEKTROMETRIJOS (XRF) DUOMENŲ KOREGAVIMO MOBILIOJI APLIKACIJA 
        ### Aprašymas
        Ši aplikacija koreguoja matavimų vertes, gautas tiriant pXRF ir ED-XRF spektrometrais, į etalonines vertes, naudodama sukurtą prognostinį modelį.
        ### Kaip naudoti?
        Įkelti failą (CSV, XLSX, XLS) su duotomis dirbinio sudėties reikšmėmis (%): įkelti failą paspaudžiant "Browse file".
    """)

    ## Displays the user input features
    streamlit.write("""
        ### Kaip turėtų atrodyti įvesties duomenys?
        - Žiūrėti pateiktą pavyzdį apačioje. Tai yra reikalaujamas įvesties failo formatas.
        - Galite pateikti vieną ar daugiau įvesties eilučių.
        - **Metodas** stulpelio galimos reikšmės: **pXRF** arba **ED-XRF**.
        - **Mėginio tipas** stulpelio galimos reikšmės: **Kietas paviršius** arba **Drožlės**.
    """)
    streamlit.write(example_df)

    ## Read input data and use model to make predictions
    ### Collects user input features into dataframe
    streamlit.subheader("Vartotojo įkeltos reikšmės")
    streamlit.write("""
        - Jeigu įvesties failas nenurodytas, įvesties duomenys ir prognozės yra generuojamos naudojant pavyzdinių duomenų rinkinį.
    """)
    streamlit.write(input_df)

    ### Predictions

    # todo: add labelencoder
    input_df["Metodas"] = numpy.where(input_df['Metodas'] == "pXRF", 1, 0)
    input_df["Mėginio tipas"] = numpy.where(input_df['Mėginio tipas'] == "Kietas paviršius", 1, 0)

    predictions = utils.predict(model, input_df)
    streamlit.subheader("Koreguotos reikšmės pritaikius prognostinį modelį")
    streamlit.write(predictions)

    ### Allow to download prediction data
    @streamlit.experimental_memo
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(predictions)

    streamlit.download_button(
        "Paspausk šį mygtuką, kad atsisiųsti gautas reikšmes",
        csv,
        OUTPUT_FILE_NAME,
        "text/csv",
        key='download-csv'
    )
