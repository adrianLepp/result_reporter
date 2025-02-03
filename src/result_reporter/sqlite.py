import sqlalchemy as sa
from sqlalchemy import ForeignKey
from sqlalchemy.orm import sessionmaker, Mapped, mapped_column, declarative_base
from typing import List
import json
from datetime import datetime
#-------------------------------------------------------------------------------
from result_reporter.type import Model_Config, Simulation_Config, System_Description, Data

INIT = False
CONFIG_FILE = 'config.json'
Base = declarative_base()

def init():
    global db, Session, Base, INIT
    if not INIT:
        with open(CONFIG_FILE,"r") as f:
            config = json.load(f)
        dbName = config['db_file']
        db = sa.create_engine('sqlite:///' + dbName)
        Session = sessionmaker(bind=db)
        INIT = True

# ------------------------------------------------------------------------------------
# Table Classes
# ------------------------------------------------------------------------------------

class SystemDescription(Base):
    '''
    id, system, states, parameters
    '''
    __tablename__ = 'system_description'
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True, unique=True)
    system: Mapped[str] = mapped_column(unique=True)
    states: Mapped[str]
    parameters: Mapped[str]

    def __repr__(self):
        return f"SystemDescription(id={self.id}, system={self.system}, states={self.states}, parameters={self.parameters}))"

class ModelConfig(Base):
    '''
    id, system, init_state, system_param, t_start, t_end, dt
    '''
    __tablename__ = 'model_config'
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True, unique=True)
    system: Mapped[str]  = mapped_column(ForeignKey('system_description.system'))
    init_state: Mapped[str]
    system_param: Mapped[str]
    t_start: Mapped[float] 
    t_end: Mapped[float] 
    dt: Mapped[float] 

    def __repr__(self):
        return f"ModelConfig(id={self.id}, system={self.system}, init_state={self.init_state}, system_param={self.system_param}, t_start={self.t_start}, t_end={self.t_end}, dt={self.dt})"

class SimulationConfig(Base):
    '''
    id, model_id, system, init_state, system_param, t_start, t_end, dt, rms    
    '''
    __tablename__ = 'simulation_config'
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True, unique=True)
    system: Mapped[str]  = mapped_column(ForeignKey('system_description.system'))
    model_id: Mapped[int] = mapped_column(ForeignKey('model_config.id'))
    init_state: Mapped[str]
    system_param: Mapped[str]
    t_start: Mapped[float] 
    t_end: Mapped[float] 
    dt: Mapped[float] 
    rms: Mapped[str]
    date: Mapped[str]

    def __repr__(self):
        #return f"SimulationConfig(id={self.id}, system={self.system}, model_id={self.model_id} init_state={self.init_state}, system_param={self.system_param}, t_start={self.t_start}, t_end={self.t_end}, dt={self.dt})"
        return f"SimulationConfig(id={self.id}, system={self.system}, model_id={self.model_id} date={self.date})"
    
class Training_Data(Base):
    '''
    model_id, t, x1, x2, x3, x4, x5    
    '''
    __tablename__ = 'training_data'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    #id = Column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey('model_config.id'))
    #system: Mapped[str]  = mapped_column(ForeignKey('system_description.system'))
    t: Mapped[float] 
    f1 : Mapped[float] = mapped_column(default=None, nullable=True)
    f2 : Mapped[float] = mapped_column(default=None, nullable=True)
    f3 : Mapped[float] = mapped_column(default=None, nullable=True)
    f4 : Mapped[float] = mapped_column(default=None, nullable=True)
    f5 : Mapped[float] = mapped_column(default=None, nullable=True)

    def __repr__(self): #id={self.id}, 
        return f"Training_Data(t={self.t}, f1={self.f1}, f2={self.f2}, f3={self.f3}, f4={self.f4}, f5={self.f5})"

class Simulation_Data(Base):
    '''
    simulation_id, t, x1, x2, x3, x4, x5    
    '''
    __tablename__ = 'simulation_data'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    simulation_id: Mapped[int] = mapped_column(ForeignKey('simulation_config.id'))
    #model_id: Mapped[int] = mapped_column(ForeignKey('model_config.id'))
    #system: Mapped[str]  = mapped_column(ForeignKey('system_description.system'))
    t: Mapped[float] 
    f1 : Mapped[float] = mapped_column(default=None, nullable=True)
    f2 : Mapped[float] = mapped_column(default=None, nullable=True)
    f3 : Mapped[float] = mapped_column(default=None, nullable=True)
    f4 : Mapped[float] = mapped_column(default=None, nullable=True) 
    f5 : Mapped[float] = mapped_column(default=None, nullable=True)

    def __repr__(self):#id={self.id}, 
        return f"Simulation_Data(t={self.t}, f1={self.f1}, f2={self.f2}, f3={self.f3}, f4={self.f4}, f5={self.f5})"
    
class Reference_Data(Base):
    '''
    simulation_id, type, t, x1, x2, x3, x4, x5
    '''
    __tablename__ = 'reference_data'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    simulation_id: Mapped[int] = mapped_column(ForeignKey('simulation_config.id'))
    type: Mapped[str] = mapped_column(default=None, nullable=True)

    t: Mapped[float] 
    f1 : Mapped[float] = mapped_column(default=None, nullable=True)
    f2 : Mapped[float] = mapped_column(default=None, nullable=True)
    f3 : Mapped[float] = mapped_column(default=None, nullable=True)
    f4 : Mapped[float] = mapped_column(default=None, nullable=True) 
    f5 : Mapped[float] = mapped_column(default=None, nullable=True)

    def __repr__(self):#id={self.id}, 
        return f"Reference_Data(t={self.t}, f1={self.f1}, f2={self.f2}, f3={self.f3}, f4={self.f4}, f5={self.f5})"
    
# ------------------------------------------------------------------------------------
# add Data
# ------------------------------------------------------------------------------------

def commitData(entry):
    Base.metadata.create_all(db)
    with Session() as session:
        session.add(entry)
        session.commit()

def add_systemDescription(system:str, states:List[str], parameters:List[str]):
    init()
    _states = list_to_str(states, dtype=str)
    _parameters = list_to_str(parameters, dtype=str)
    system_description = SystemDescription(system=system, states=_states, parameters=_parameters)
    commitData(system_description)

def add_modelConfig(id:int, system:str, init_state:List[float], system_param:List[float], t_start:float, t_end:float, dt:float):
    init()
    _init_state = list_to_str(init_state, dtype=float)
    _system_param = list_to_str(system_param, dtype=float)
    model_config = ModelConfig(id=id, system=system, init_state=_init_state, system_param=_system_param, t_start=t_start, t_end=t_end, dt=dt)
    commitData(model_config)
    

def add_simulationConfig(id:int, model_id:int, system:str, init_state:List[float], system_param:List[float], t_start:float, t_end:float, dt:float, rms:List[float]):
    init()
    today = datetime.today().strftime('%d.%m.%Y')
    _init_state = list_to_str(init_state, dtype=float)
    _system_param = list_to_str(system_param, dtype=float)
    _rms = list_to_str(rms, dtype=float)
    simulation_config = SimulationConfig(id=id, model_id=model_id, system=system, init_state=_init_state, system_param=_system_param, t_start=t_start, t_end=t_end, dt=dt, rms=_rms, date=today)
    commitData(simulation_config)

def data_to_dict(time, states):
    data = []
    for i in range(len(time)):
        state_dict = {}
        for l in range(len(states[i])):
            state_dict['f'+str(l+1)] = states[i][l]
        data.append(state_dict)
    return data

def addBulkData(bulk_list):
    with Session() as session:
        session.bulk_save_objects(bulk_list)
        session.commit()

def add_training_data(model_id:int, time, states):
    init()
    data = data_to_dict(time, states)

    Base.metadata.create_all(db)
    # bulk insert
    bulk_list = []
    for i in range(len(time)):
        bulk_list.append(Training_Data(model_id=model_id, t=time[i], **data[i]))

    addBulkData(bulk_list)

def add_simulation_data(simulation_id:int, time, states):
    init()
    data = data_to_dict(time, states)
    # bulk insert
    bulk_list = []
    for i in range(len(time)):
        bulk_list.append(Simulation_Data(simulation_id=simulation_id, t=time[i], **data[i]))

    addBulkData(bulk_list)

def add_reference_data(simulation_id:int, type: str, time, states):
    init()
    data = data_to_dict(time, states)
    # bulk insert
    bulk_list = []
    for i in range(len(time)):
        bulk_list.append(Reference_Data(simulation_id=simulation_id, type=type, t=time[i], **data[i]))

    addBulkData(bulk_list)

# ------------------------------------------------------------------------------------
# get Data
# ------------------------------------------------------------------------------------

def convert_config_values(config):
    config['init_state'] =  str_to_list(config['init_state'], dtype=float)
    config['system_param'] = str_to_list(config['system_param'], dtype=float)
    if 'rms' in config and config['rms'] != '':
        config['rms'] = str_to_list(config['rms'], dtype=float)
    else:
        config['rms'] = []
    return config

def list_to_str(lst: List, dtype: type = float) -> str:
    return ', '.join(map(str, map(dtype, lst)))

def str_to_list(s: str, dtype: type = float) -> List:
    return list(map(dtype, s.split(',')))

# def lst_str(lst: List[float]) -> str:
#     return ', '.join(map(str, lst))

# def str_lst(s: str) -> List[float]:
#     return list(map(float, s.split(',')))

def convert_data(data) -> Data:
    '''
    FIXME: this function is very ugly but element is not subscriptable
    - could it be faster to transform every element to a dict?
    '''
    state_n = 5
    time = []
    states = []

    data_dict = {
        'time': [],
        'f1': [],
        'f2': [],
        'f3': [],
        'f4': [],
        'f5': []
    }

    for element in data:
        state_row = []
        time.append(element.t)

        data_dict['time'].append(element.t)
        data_dict['f1'].append(element.f1)
        data_dict['f2'].append(element.f2)
        data_dict['f3'].append(element.f3)
        data_dict['f4'].append(element.f4)
        data_dict['f5'].append(element.f5)
        
    #     for i in range(0, state_n):
    #         if element[f'f{i+1}'] is not None:
    #             state_row.append(element[f'f{i+1}'])
    #         else:
    #             state_n = i
    #             break
    #     states.append(state_row)

    # return time, states
    if len(data_dict['time']) == 0:
            return None

    if data_dict['f5'][0] is None:
        del data_dict['f5']
    if data_dict['f4'][0] is None:
        del data_dict['f4']
    if data_dict['f3'][0] is None:
        del data_dict['f3']
    if data_dict['f2'][0] is None:
        del data_dict['f2']
    if data_dict['f1'][0] is None:
        del data_dict['f1']
    return data_dict

def get_system_description(id:int=None, system:str=None) -> List[System_Description] | System_Description:
    init()
    with Session() as session:
        if id is not None:
            config = session.get(SystemDescription, id).__dict__
            config['states'] = str_to_list(config['states'], dtype=str)
            config['parameters'] = str_to_list(config['parameters'], dtype=str)
            return config
        elif system is not None: 
            statement = sa.select(SystemDescription).filter_by(system=system)
            config = session.scalars(statement).one().__dict__
        
            config['states'] = str_to_list(config['states'], dtype=str)
            config['parameters'] = str_to_list(config['parameters'], dtype=str)
            return config
        else:
            result = session.query(SystemDescription).all()
            return [element.__dict__ for element in result]

def get_model_config(id:int=None, system:str=None)->List[Model_Config] | Model_Config:
    init()
    with Session() as session:
        if id is not None:
            #return session.query(ModelConfig).filter_by(id=id)
            config = session.get(ModelConfig, id).__dict__
            return convert_config_values(config)
        else:
            if system is not None:
                statement = sa.select(ModelConfig).filter_by(system=system)
                result = session.scalar(statement).all() 
                #with session.execute() one could call .as_dict() on the result
            else:
                result =  session.query(ModelConfig).all()

            return [convert_config_values(element.__dict__) for element in result]
    


def get_simulation_config(id:int=None, model_id:int=None, system:str=None)->List[Simulation_Config] | Simulation_Config:
    init()
    with Session() as session:
        if id is not None:
            config =  session.get(SimulationConfig, id).__dict__
            return convert_config_values(config)
        else:
            if model_id is not None:
                statement = sa.select(SimulationConfig).filter_by(model_id=model_id)
                result =  session.scalars(statement).all()
            
            elif system is not None:
                statement = sa.select(SimulationConfig).filter_by(system=system)
                result =  session.scalars(statement).all()
            else:
                result = session.query(SimulationConfig).all()
            
            return [convert_config_values(element.__dict__) for element in result]

def get_training_data(id:int):
    init()
    with Session() as session:
        statement = sa.select(Training_Data).filter_by(model_id=id)
        data = session.scalars(statement).all()
    return convert_data(data)

def get_simulation_data(id:int):
    init()
    with Session() as session:
        statement = sa.select(Simulation_Data).filter_by(simulation_id=id)
        data = session.scalars(statement).all()
    return convert_data(data)

def get_reference_data(id:int, type:str=None):
    init()
    with Session() as session:
        if type is None:
            statement = sa.select(Reference_Data).filter_by(simulation_id=id)
        else:
            statement = sa.select(Reference_Data).filter_by(simulation_id=id, type=type)
        data = session.scalars(statement).all()
    return convert_data(data)
    



