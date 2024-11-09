import os
import json
import pandas as pd
import numpy as np
from typing import List

def saveDataToCsv(simName:str, data:dict, overwrite:bool=False):
    fileName = simName +  '.csv'

    if os.path.exists(fileName) and not overwrite:
        raise FileExistsError(f"The file {fileName} already exists.")
    
    df = pd.DataFrame(data)
    df.to_csv(fileName, index=False)


def saveSettingsToJson(simName:str, settings:dict, overwrite:bool=False):
    fileName = simName + '.json'
    if os.path.exists(fileName) and not overwrite:
        raise FileExistsError(f"The file {fileName} already exists.")

    with open(fileName,"w") as f:
        json.dump(settings, f)
