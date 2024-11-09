import pandas as pd
from typing import List
import json

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