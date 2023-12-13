
# imports
import json
from package.resources.utility import createFolder,produce_name_datetime,save_object
from package.resources.run import generate_data,parallel_run

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
         ) -> str: 

    f = open(BASE_PARAMS_LOAD)
    base_params = json.load(f)

    root = "culture_compare_time_series"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    culture_list = ["dynamic_culturally_determined_weights","dynamic_socially_determined_weights","static_culturally_determined_weights", "fixed_preferences"]
    params_list = []
    for i in culture_list:
        base_params["alpha_change_state"] =  i
        params_list.append(
                base_params.copy()
            )  

    Data_list = parallel_run(params_list)

    createFolder(fileName)

    save_object(culture_list,fileName + "/Data", "culture_list")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(Data_list, fileName + "/Data", "Data_list")

    return fileName

if __name__ == '__main__':
    fileName_Figure_1 = main(
        BASE_PARAMS_LOAD = "package/constants/base_params_culture.json",
)
