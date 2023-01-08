import pandas
import xgboost

def predict(model: xgboost.XGBRegressor, input_data: pandas.DataFrame) -> pandas.DataFrame:
    prediction = model.predict(input_data)
    return pandas.DataFrame(prediction, columns=input_data.columns[:-2])
