
import os.path as path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from result_reporter.plotConfig import set_size
import pandas as pd
from result_reporter.data_loader import loadDataFromCSV, loadParamfromJson
from typing import List

print(plt.style.available)
print("Your style sheets are located at: {}".format(path.join(mpl.__path__[0], 'mpl-data', 'stylelib')))

#https://jwalton.info/Embed-Publication-Matplotlib-Latex/   


#plt.style.use('seaborn')
try:
    plt.style.use('seaborn-v0_8-paper')
    plt.style.use('tex')
except:
    print('style not found')
# in LaTex show the textWidth with '\the\textwidth'
textWidth= 469.4704 #TODO: environment variable

#---------------------------------------
fileFolder = '../data/'
imgFolder = '../img/'
fileFormat = 'pdf'


def saveSingleImageToPdf(folder, fileName, header:List[str], xLabel='Time [s]', yLabel='y', title=None):
    rows = 1
    cols = 1
    scale = 1
    frac = 1

    data = loadDataFromCSV(folder, fileName, header)

    fig, ax = plt.subplots(rows, cols, figsize=set_size(scale*textWidth, frac, (rows, cols)))

    try:
        time_idx = header.index('time')
        time = data[time_idx]
    except:
        time_idx = -1

    for i in range(len(header)):
        if time_idx != i:
            ax.plot(time, data[i], label=header[i])

    ax.legend()
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.grid(True)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(imgFolder + fileName + '.pdf', format=fileFormat, bbox_inches='tight')


def create_report_plot(training_data, sim_data, header:List[str], xLabel='Time [s]', yLabel='y', title=None):
    rows = 1
    cols = 1
    scale = 1
    frac = 1

    fig, ax = plt.subplots(rows, cols, figsize=set_size(scale*textWidth, frac, (rows, cols)))



    for i in range(len(header)):
        ax.plot(training_data['time'], training_data[f'f{i+1}'], label=header[i], linestyle='--')
        ax.plot(sim_data['time'], sim_data[f'f{i+1}'], label='train ' + header[i])
        

    ax.legend()
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.grid(True)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig

def save_plot_to_pdf(fig, fileName):
    fig.savefig(imgFolder + fileName + '.pdf', format=fileFormat, bbox_inches='tight')
