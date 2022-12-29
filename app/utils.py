import pandas
import xgboost

def predict(model: xgboost.XGBRegressor, input_data: pandas.DataFrame) -> pandas.DataFrame:
    prediction = model.predict(input_data)
    prediction_df = input_data.copy()
    prediction_df.iloc[:, :-2] = pandas.DataFrame(prediction)
    return prediction_df
