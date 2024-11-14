from typing import TypedDict, List, Optional

class System_Description(TypedDict):
    id:int
    name:str
    states:List[str]
    parameters:List[str]

class Model_Config(TypedDict):
    id:int
    system:str
    t_start:float
    t_end:float
    dt:float
    init_state:List[float]
    system_param:List[float]

class Simulation_Config(TypedDict):
    id:int
    model_id:int
    system:str
    t_start:float
    t_end:float
    dt:float
    init_state:List[float]
    system_param:List[float]
    rms:List[float]
    date:str

class Data(TypedDict):
    time:List[float]
    f1:Optional[List[float]]
    f2:Optional[List[float]]
    f3:Optional[List[float]]
    f4:Optional[List[float]]
    f5:Optional[List[float]]