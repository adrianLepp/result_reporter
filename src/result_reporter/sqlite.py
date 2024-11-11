import sqlalchemy as sa
from sqlalchemy import ForeignKey
from sqlalchemy.orm import sessionmaker, Mapped, mapped_column, declarative_base
from typing import List
from type import Model_Config, Simulation_Config

INIT = False
dbName = '../db/lodegp_test4.db'
Base = declarative_base()

def init():
    global db, Session, Base, INIT
    if not INIT:
        db = sa.create_engine('sqlite:///' + dbName)
        Session = sessionmaker(bind=db)
        INIT = True

# db = sa.create_engine('sqlite:///' + dbName)
# Session = sessionmaker(bind=db)
# Base = declarative_base()

class ModelConfig(Base):
    '''
    id, system, init_state, system_param, t_start, t_end, dt
    '''
    __tablename__ = 'model_config'
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True, unique=True)
    system: Mapped[str] 
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
    system: Mapped[str]
    model_id: Mapped[int] = mapped_column(ForeignKey('model_config.id'))
    init_state: Mapped[str]
    system_param: Mapped[str]
    t_start: Mapped[float] 
    t_end: Mapped[float] 
    dt: Mapped[float] 
    rms: Mapped[float]

    def __repr__(self):
        return f"SimulationConfig(id={self.id}, system={self.system}, model_id={self.model_id} init_state={self.init_state}, system_param={self.system_param}, t_start={self.t_start}, t_end={self.t_end}, dt={self.dt})"
    

def add_modelConfig(id:int, system:str, init_state:List[float], system_param:List[float], t_start:float, t_end:float, dt:float):
    init()
    model_config = ModelConfig(id=id, system=system, init_state=lst_str(init_state), system_param=lst_str(system_param), t_start=t_start, t_end=t_end, dt=dt)
    commitData(model_config)
    

def add_simulationConfig(id:int, model_id:int, system:str, init_state:List[float], system_param:List[float], t_start:float, t_end:float, dt:float, rms:float):
    init()
    #Base.metadata.create_all(db)
    simulation_config = SimulationConfig(id=id, model_id=model_id, system=system, init_state=lst_str(init_state), system_param=lst_str(system_param), t_start=t_start, t_end=t_end, dt=dt, rms=rms)
    commitData(simulation_config)
    # with Session() as session:
    #     session.add(simulation_config)
    #     session.commit()

def commitData(entry):
    Base.metadata.create_all(db)
    with Session() as session:
        session.add(entry)
        session.commit()

class Training_Data(Base):
    '''
    model_id, system,  t, x1, x2, x3, x4, x5    
    '''
    __tablename__ = 'training_data'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    #id = Column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey('model_config.id'))
    system: Mapped[str]
    t: Mapped[float] 
    f1 : Mapped[float] = mapped_column(default=None, nullable=True)
    f2 : Mapped[float] = mapped_column(default=None, nullable=True)
    f3 : Mapped[float] = mapped_column(default=None, nullable=True)
    f4 : Mapped[float] = mapped_column(default=None, nullable=True)
    f5 : Mapped[float] = mapped_column(default=None, nullable=True)

    def __repr__(self): #id={self.id}, 
        return f"Training_Data(system={self.system}, model_id={self.model_id}, t={self.t}, f1={self.f1}, f2={self.f2}, f3={self.f3}, f4={self.f4}, f5={self.f5})"

class Simulation_Data(Base):
    '''
    model_id, system,  simulation_id, t, x1, x2, x3, x4, x5    
    '''
    __tablename__ = 'simulation_data'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    simulation_id: Mapped[int] = mapped_column(ForeignKey('simulation_config.id'))
    model_id: Mapped[int] = mapped_column(ForeignKey('model_config.id'))
    system: Mapped[str]
    t: Mapped[float] 
    f1 : Mapped[float] = mapped_column(default=None, nullable=True)
    f2 : Mapped[float] = mapped_column(default=None, nullable=True)
    f3 : Mapped[float] = mapped_column(default=None, nullable=True)
    f4 : Mapped[float] = mapped_column(default=None, nullable=True) 
    f5 : Mapped[float] = mapped_column(default=None, nullable=True)

    def __repr__(self):#id={self.id}, 
        return f"Simulation_Data(system={self.system}, simulation_id={self.simulation_id}, model_id={self.model_id}, t={self.t}, f1={self.f1}, f2={self.f2}, f3={self.f3}, f4={self.f4}, f5={self.f5})"

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

def add_training_data(model_id:int, system:str,  time, states):#id:int
    init()
    data = data_to_dict(time, states)

    Base.metadata.create_all(db)
    # bulk insert
    bulk_list = []
    for i in range(len(time)):
        bulk_list.append(Training_Data(model_id=model_id, system=system,  t=time[i], **data[i]))

    addBulkData(bulk_list)

def add_simulation_data(simulation_id:int, model_id:int, system:str,  time, states):#id:int, 
    init()
    data = data_to_dict(time, states)
    # bulk insert
    bulk_list = []
    for i in range(len(time)):
        bulk_list.append(Simulation_Data(simulation_id=simulation_id, model_id=model_id, system=system, t=time[i], **data[i]))

    addBulkData(bulk_list)

# def add_data(type:str, id:int, time, states, system:str, model_id:int, simulation_id:int=None):
#     #TODO how do i format the data
#     #https://docs.sqlalchemy.org/en/20/changelog/whatsnew_20.html#change-6047
#     #https://stackoverflow.com/questions/3659142/bulk-insert-with-sqlalchemy-orm
#     data = {'time': time}
#     for i in range(len(states[1])):
#         data[f'f{i+1}'] = states[:, i]
    
#     if type == 'training':
#         entry = Training_Data(**data)
#     elif type == 'simulation':
#         entry = Simulation_Data(id, system, model_id, simulation_id, time, **data)


#     # bulk insert
#     bulk_list = []
#     for i in range(len(time)):
#         bulk_list.append(Training_Data(id, system, model_id, time[i], **data[i]))

#     with Session() as session:
#         session.bulk_save_objects(bulk_list)
#         session.commit()

    # add
    # with Session() as session:
    #     for i in range(len(time)):
    #         session.add(Training_Data(id, system, model_id, time[i], **data[i]))
    #     session.commit()

    # # add all
    # bulk_list = []
    # for i in range(len(time)):
    #     bulk_list.append(Training_Data(id, system, model_id, time[i], **data[i]))

    # with Session() as session:
    #     session.add_all(bulk_list)
    #     session.commit()

    


    
    #commitData(entry)

# ----------------------------
# getters
# ----------------------------

def convert_config_values(config):
    config['init_state'] =  str_lst(config['init_state'])
    config['system_param'] = str_lst(config['system_param'])
    return config

def lst_str(lst: List[float]) -> str:
    return ', '.join(map(str, lst))

def str_lst(s: str) -> List[float]:
    return list(map(float, s.split(',')))

def convert_data(data):
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
            return session.get(SimulationConfig, id).__dict__
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
    



