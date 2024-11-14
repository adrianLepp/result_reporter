import matplotlib
from pylatex import Document, Figure, NoEscape, Section, Itemize, Subsection
import json
# ---------------------------------------
from result_reporter.latex_exporter import create_report_plot
from result_reporter.sqlite import get_model_config, get_simulation_config, get_training_data, get_simulation_data, get_system_description
from result_reporter.type import Model_Config, Simulation_Config, Data, System_Description

NUMBERING = False
CONFIG_FILE = 'config.json'

figureWidth = r"0.8\textwidth"

def parse_float(value:float):
    return f"{value:.2e}"

def load_data(sim_id):
    simulation_config:Simulation_Config = get_simulation_config(sim_id)
    model_id = simulation_config['model_id']
    system_description:System_Description = get_system_description(system=simulation_config['system'])
    model_config:Model_Config = get_model_config(model_id)
    training_data:Data = get_training_data(model_id)
    simulation_data:Data = get_simulation_data(sim_id)

    return simulation_config, model_config, system_description, training_data, simulation_data

def get_report_folder():
    with open(CONFIG_FILE,"r") as f:
        config = json.load(f)
    return config['report_folder']

def create_report(sim_id, info_text:str=None):
    matplotlib.use("Agg")  # Not to use X server. For TravisCI.
    simulation_config, model_config, system_description,training_data, simulation_data = load_data(sim_id)
    dir = get_report_folder()


    filename = f"{dir}/report_{simulation_config['id']}_{model_config['system']}"
    title = "Simulation" + ' ' + str(simulation_config['id']) + ' - '  + 'Model:  ' + simulation_config['system']
    
    geometry_options = {"right": "2cm", "left": "2cm"}
    doc = Document(filename, geometry_options=geometry_options)

    

    with doc.create(Section(
        title, numbering=NUMBERING)):

        if info_text is not None:
            doc.append(info_text)
        
        with doc.create(Itemize()) as itemize:
            itemize.add_item(f"System: {model_config['system']}, id: {system_description['id']}")
            itemize.add_item(f"Model-Id: {model_config['id']}")
            itemize.add_item(f"Simulation Date: {simulation_config['date']}")
        
        with doc.create(Figure(position="htbp")) as plot:
            create_report_plot(training_data, simulation_data, system_description['states'])
            plot.add_plot(width=NoEscape(figureWidth))
            plot.add_caption("Simulation results")

        #doc.append('Simulation-Parameter')

        with doc.create(Subsection("Parameter", numbering=NUMBERING)):
            with doc.create(Itemize()) as itemize:
                itemize.add_item(f"Time Interval: [{str(simulation_config['t_start'])}, {str(simulation_config['t_end'])}]")
                itemize.add_item("dt: " + str(simulation_config['dt']))
                # add system parameter
                itemize.add_item(f"System Param: [")
                for i, state in enumerate(system_description['states']):
                    doc.append(f"{state}, ")
                doc.append("] = [")
                for i, state in enumerate(system_description['parameters']):
                    doc.append(f"{parse_float(model_config['system_param'][i])}, ")
                doc.append("]")
                # add initial state
                itemize.add_item(f"Initial State: [")
                for i, state in enumerate(system_description['states']):
                    doc.append(f"{state}, ")
                doc.append("] = [")
                for i, state in enumerate(system_description['states']):
                    doc.append(f"{parse_float(model_config['init_state'][i])}, ")
                doc.append("]")
                
        with doc.create(Subsection("Metrics", numbering=NUMBERING)):
            with doc.create(Itemize()) as itemize:
                rms = [parse_float(item) for item in simulation_config['rms']]
                itemize.add_item(f"RMS: {simulation_config['rms']}")

    doc.generate_pdf(clean_tex=False)
