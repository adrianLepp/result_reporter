import matplotlib
from pylatex import Document, Figure, NoEscape, Section, Itemize

matplotlib.use("Agg")  # Not to use X server. For TravisCI.
import matplotlib.pyplot as plt  # noqa
from latex_exporter import saveSingleImageToPdf, create_report_plot
from data_loader import loadDataFromCSV, loadParamfromJson
from sqlite import get_model_config, get_simulation_config, get_training_data, get_simulation_data
from type import Model_Config, Simulation_Config, Data

fileFolder = '../data/'


figureWidth = r"0.8\textwidth"

# def main(fname, width, *args, **kwargs):
#     geometry_options = {"right": "2cm", "left": "2cm"}
#     doc = Document(fname, geometry_options=geometry_options)

#     doc.append("Introduction.")

#     with doc.create(Section("I am a section")):
#         doc.append("Take a look at this beautiful plot:")

#         with doc.create(Figure(position="htbp")) as plot:
#             plot.add_plot(width=NoEscape(width), *args, **kwargs)
#             plot.add_caption("I am a caption.")

#         doc.append("Created using matplotlib.")

#     doc.append("Conclusion.")

#     doc.generate_pdf(clean_tex=False)

def load_data(sim_id):
    simulation_config:Simulation_Config = get_simulation_config(sim_id)
    model_id = simulation_config['model_id']
    model_config:Model_Config = get_model_config(model_id)
    training_data:Data = get_training_data(model_id)
    simulation_data:Data = get_simulation_data(sim_id)

    return simulation_config, model_config, training_data, simulation_data

def create_report(sim_id):
    simulation_config, model_config, training_data, simulation_data = load_data(sim_id)

    filename = str(simulation_config['id']) + '_' + model_config['system']
    title = "Simulation" + ' ' + str(simulation_config['id']) + ' - '  + 'Model:  ' + simulation_config['system']
    
    geometry_options = {"right": "2cm", "left": "2cm"}
    doc = Document('report_' + filename, geometry_options=geometry_options)

    #doc.append(sim_settings['date'])

    with doc.create(Section(
        title, numbering=False)):
        # doc.append("This is a simulation:")
        with doc.create(Itemize()) as itemize:
            itemize.add_item("System: " + model_config['system'])
            itemize.add_item("Model-Id: " + str(model_config['id']))
            
        with doc.create(Figure(position="htbp")) as plot:
            create_report_plot(training_data, simulation_data, ['f1', 'f2', 'f3', 'u'])
            plot.add_plot(width=NoEscape(figureWidth))
            plot.add_caption("Simulation results")

        doc.append('Simulations-Parameter')
        with doc.create(Itemize()) as itemize:
            itemize.add_item(f"Time Interval: [{str(simulation_config['t_start'])}, {str(simulation_config['t_end'])}]")
            itemize.add_item("dt: " + str(simulation_config['dt']))
            itemize.add_item("Initial-State: " + str(simulation_config['init_state']))
            itemize.add_item("System-Parameter: " + str(simulation_config['system_param']))
                
        doc.append('Metrics')
        with doc.create(Itemize()) as itemize:
            itemize.add_item("RMS: " + str(simulation_config['rms']))


        

        #doc.append("Created using matplotlib.")

    #doc.append("Conclusion.")

    doc.generate_pdf(clean_tex=False)

def plot_data():
    pass

def createReport_deprecated(fileName, sim_settings):
    geometry_options = {"right": "2cm", "left": "2cm"}
    doc = Document('report_' + fileName, geometry_options=geometry_options)


    doc.append(sim_settings['date'])

    with doc.create(Section(
        "Simulation" + ' ' + str(sim_settings['id']) + ' - ' + sim_settings['model'] + ' ' + sim_settings['system'], 
        numbering=False)):
        # doc.append("This is a simulation:")

        with doc.create(Figure(position="htbp")) as plot:
            plot.add_plot(width=NoEscape(figureWidth))
            plot.add_caption("Simulation results")
        
        with doc.create(Itemize()) as itemize:
            itemize.add_item("RMS: " + str(sim_settings['rms']))


        

        #doc.append("Created using matplotlib.")

    #doc.append("Conclusion.")

    doc.generate_pdf(clean_tex=False)


if __name__ == "__main__":
    create_report(4)
#     x = [0, 1, 2, 3, 4, 5, 6]
#     y = [15, 2, 7, 1, 5, 6, 9]

#     plt.plot(x, y)

# name='0_lodegp_bipendulum'
# saveSingleImageToPdf(fileFolder, name, ['time', 'f1', 'f2', 'f3'], title='Simulation data')
# settings = loadParamfromJson(fileFolder, name)
# createReport(name, settings)