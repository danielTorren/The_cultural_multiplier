# imports
from package.resources.run import generate_data
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)
from package.plotting_data import example_timeseries_plot
import pyperclip
import json

def main(
   BASE_PARAMS_LOAD = ""
) -> str: 
    
    f = open(BASE_PARAMS_LOAD)
    base_params = json.load(f)

    root = "example_timerseries"
    fileName = produce_name_datetime(root)
    pyperclip.copy(fileName)
    print("fileName:", fileName)

    #low carbon price
    base_params["carbon_price_increased"] = 0
    Data_no = generate_data(base_params)  # run the simulation

    #high carbon price
    base_params["carbon_price_increased"] = 0.15
    Data_high = generate_data(base_params)  # run the simulation

    #print(Data.average_identity)

    createFolder(fileName)
    save_object(Data_no, fileName + "/Data", "Data_no")
    save_object(Data_high, fileName + "/Data", "Data_high")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == '__main__':
    
    fileName = main(BASE_PARAMS_LOAD = "package/constants/base_params_timeseries.json")

    RUN_PLOT = 1

    if RUN_PLOT:
        example_timeseries_plot.main(fileName = fileName)
