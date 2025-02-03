import pandas as pd
from typing import List
import json

from result_reporter.sqlite import get_model_config, get_simulation_config, get_training_data, get_simulation_data, get_system_description, get_reference_data
from result_reporter.type import Model_Config, Simulation_Config, Data, System_Description

def loadDataFromCSV(folder, fileName, header:List[str]):
    df = pd.read_csv(folder + fileName + '.csv')
    data = []
    for i in range(len(header)):
        data.append(df[header[i]])

    return data

def loadParamfromJson(folder, fileName):
    with open(folder + fileName + '.json') as f:
        data = json.load(f)
    return data

def load_data(sim_id:int):
    simulation_config:Simulation_Config = get_simulation_config(sim_id)
    model_id = simulation_config['model_id']
    system_description:System_Description = get_system_description(system=simulation_config['system'])
    model_config:Model_Config = get_model_config(model_id)
    training_data:Data = get_training_data(model_id)
    simulation_data:Data = get_simulation_data(sim_id)

    reference_data:Data = get_reference_data(sim_id)#,type='nonlinear'

    return simulation_config, model_config, system_description, training_data, simulation_data, reference_data