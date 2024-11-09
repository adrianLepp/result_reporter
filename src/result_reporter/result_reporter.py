import matplotlib
from pylatex import Document, Figure, NoEscape, Section, Itemize

matplotlib.use("Agg")  # Not to use X server. For TravisCI.
import matplotlib.pyplot as plt  # noqa
from latex_exporter import saveSingleImageToPdf 
from data_loader import loadDataFromCSV, loadParamfromJson

fileFolder = '../data/'


figureWidth = r"0.5\textwidth"

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

def createReport(fileName, sim_settings):
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


# if __name__ == "__main__":
#     x = [0, 1, 2, 3, 4, 5, 6]
#     y = [15, 2, 7, 1, 5, 6, 9]

#     plt.plot(x, y)

name='0_lodegp_bipendulum'
saveSingleImageToPdf(fileFolder, name, ['time', 'f1', 'f2', 'f3'], title='Simulation data')
settings = loadParamfromJson(fileFolder, name)
createReport(name, settings)