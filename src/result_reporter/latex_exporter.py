
import os.path as path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from result_reporter.plotConfig import set_size, CM_to_PT
import pandas as pd
from result_reporter.data_loader import loadDataFromCSV, loadParamfromJson
from typing import List
from matplotlib.legend_handler import HandlerTuple

#https://jwalton.info/Embed-Publication-Matplotlib-Latex/   



#---------------------------------------

INIT = False

fileFolder = '../data/'
imgFolder = '../img/'
fileFormat = 'pdf'

def init():
    global textWidth, Session, Base, INIT
    if not INIT:
        # in LaTex show the textWidth with '\the\textwidth'
        textWidth= 469.4704
        try:
            print(plt.style.available)
            print("Your style sheets are located at: {}".format(path.join(mpl.__path__[0], 'mpl-data', 'stylelib')))
            plt.style.use('seaborn-v0_8-paper')
            plt.style.use('tex')
            INIT = True
        except:
            print('style not found')
        

def saveSingleImageToPdf(folder, fileName, header:List[str], xLabel='Time [s]', yLabel='y', title=None):
    init()
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


def create_report_plot(training_data, sim_data, header:List[str], xLabel='Time [s]', yLabel='y', reference_data=None, x_e=None, title=None):
    init()
    rows = 1
    cols = 1
    scale = 1
    frac = 1

    alpha_train = 0.5
    alpha_est = 0.7

    fig, ax1 = plt.subplots(rows, cols, figsize=set_size(scale*textWidth, frac, (rows, cols)))
    ax2 = ax1.twinx()

    if x_e is not None:
        equilibrium = [x_e, x_e]
        t_range = [training_data['time'][0], training_data['time'][-1]]


    for i in range(len(header)):
        try:
            color = f'C{i}'
            if 'u' in header[i]:
                ax2.plot(sim_data['time'], sim_data[f'f{i+1}'], label=header[i], color=color, alpha=alpha_est)
                if reference_data and len(reference_data) > 0: 
                    ax2.plot(reference_data['time'], reference_data[f'f{i+1}'], '--', label=f'{header[i]} ref', color=color, alpha=alpha_est)
                ax2.plot(training_data['time'], training_data[f'f{i+1}'], '.', label=f'{header[i]} train', color=color, alpha=alpha_train)
                
            else:
                ax1.plot(sim_data['time'], sim_data[f'f{i+1}'], label=header[i], color=color, alpha=alpha_est)
                if reference_data and len(reference_data) > 0: 
                    ax1.plot(reference_data['time'], reference_data[f'f{i+1}'], '--', label=f'{header[i]} ref', color=color, alpha=alpha_est)
                ax1.plot(training_data['time'], training_data[f'f{i+1}'], '.',label=f'{header[i]} train',  color=color, alpha=alpha_train)
        except:
            print('Some datapoints where missing: ', i)

    ax1.legend()
    ax2.legend()
    ax1.set_xlabel(xLabel)
    ax1.set_ylabel(yLabel)
    ax2.set_ylabel('u')
    ax1.grid(True)
    if title:
        ax1.set_title(title)

    fig.tight_layout()
    return fig



def create_mpc_plot(training_data, sim_data, header:List[str], xLabel='Time ($\mathrm{s})$', yLabel='Water Level ($\mathrm{m}$)', reference_data=None, x_e=None,  close_constraint=False):
    width = '8.4 cm'

    textWidth_cm = 8.4

    init()
    rows = 2
    cols = 1
    frac = 1.2
    h_scale = 0.6

    alpha_train = 1
    alpha_est = 1
    alpha_ref = 0.7

    lw_constr = 1

    fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=set_size(textWidth_cm * CM_to_PT, frac, (rows, cols), heightScale=h_scale))
    

    if x_e is not None:
        equilibrium = np.array([x_e, x_e])
        t_range = [reference_data['time'][0], reference_data['time'][-1]]

        lower = [0, 0, 0]
        upper = [0.6, 0.6,2e-4]
        lower_constraint = np.array([lower, lower])
        upper_constraint = np.array([upper, upper] )

        ref_line = ':'
        constraint_line = '--'

    lines_1_x = []
    lines_1_ref = []
    lines_2_x = []
    lines_2_ref = []

    for i in range(len(header)):
        color = f'C{i}'
        gray = 'C7'
        if 'u' in header[i]:
            lines_2_x.append(ax2.plot(reference_data['time'], reference_data[f'f{i+1}'], color=color, alpha=alpha_est)[0])
            lines_2_ref.append(ax2.plot(t_range, equilibrium[:,i], ref_line,color=color, alpha=alpha_ref)[0])
            
            if close_constraint:
                lines_2_constr = ax2.plot(t_range, equilibrium[:,i]*0.9, constraint_line, color=gray, alpha=alpha_ref, lw=lw_constr)[0]
                ax2.plot(t_range, equilibrium[:,i]*1.1, constraint_line,  color=gray, alpha=alpha_ref, lw=lw_constr)
            else:
                lines_2_constr = ax2.plot(t_range, lower_constraint[:,i], constraint_line, label='Constraints', color=gray, alpha=alpha_ref, lw=lw_constr)[0]
                #ax2.plot(t_range, upper_constraint[:,i], '--',  color=gray, alpha=alpha_ref)
        else:
            lines_1_x.append(ax1.plot(reference_data['time'], reference_data[f'f{i+1}'], color=color, alpha=alpha_est)[0]) 
            lines_1_ref.append(ax1.plot(t_range, equilibrium[:,i], ref_line, color=color, alpha=alpha_ref)[0])

            if close_constraint:
                lines_1_constr = ax1.plot(t_range, equilibrium[:,i]*0.9, constraint_line, color=gray, alpha=alpha_ref, lw=lw_constr)[0]
                ax1.plot(t_range, equilibrium[:,i]*1.1, constraint_line,  color=gray, alpha=alpha_ref, lw=lw_constr)

    labels_1 = ['$x_1$', '$x_2$', 'Reference']
    # labels_1 = ['Reference']
    labels_2 = ['$u_1$','Reference', 'Constraints']
    lines_1_ref = (lines_1_ref[0], lines_1_ref[1])

    handles_1 = [lines_1_x[0], lines_1_x[1], lines_1_ref ]
    handles_2 = [lines_2_x[0], lines_2_ref[0], lines_2_constr]

    if close_constraint:
        labels_1.append('Constraints')
        handles_1.append(lines_1_constr)

    columnspacing = 0.5
    handletextpad = 0.5
    handlelength = 1.5
    ax1.legend(handles_1, labels_1, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='upper right', fancybox=True, framealpha=0.5,ncol=4, columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength)
    ax2.legend(handles_2, labels_2, handler_map={tuple: HandlerTuple(ndivide=None)},loc='upper right', fancybox=True, framealpha=0.5,ncol=3, columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)#, wspace=-0.8 
    # ax1.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=3)#, ncol=stateN, columnspacing=columnspacing, handletextpad=handletextpad
    # ax2.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=2)
    ax2.set_xlabel(xLabel)
    ax1.set_ylabel(yLabel)
    ax2.set_ylabel('Flow rate ($\mathrm{m^3/s}$)')
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xticklabels([])
    ax1.set_xlim(t_range[0], t_range[1])
    ax2.set_xlim(t_range[0], t_range[1])
    ax1.set_ylim(0.05-0.005, 0.255)
    ax2.set_ylim(0-0.00001, 0.00015+0.00001)


    #ax2.yaxis.get_offset_text().set_position((-0.1, -0.5))
    #ax1.set_yticklabels([0.05,0.1,0.15,0.2,0.25])

    plt.plot()
    return fig

def plot_loss(loss):
    init()
    rows = 1
    cols = 1
    frac = 1
    h_scale = 0.5

    fig, ax1 = plt.subplots(rows, cols, figsize=set_size(textWidth, frac, (rows, cols), heightScale=h_scale))

    ax1.plot(loss, label='Training Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    
    ax1.legend()
    ax1.grid(True)
    fig.tight_layout()
    return fig

def plot_error(gp_error, de_error, header:List[str], xLabel='Time ($\mathrm{s})$', yLabel='Absolute error ($\mathrm{m}$)'):
    init()
    rows = 2
    cols = 1
    frac = 1
    h_scale = 0.6

    alpha_train = 1
    alpha_gp = 0.8
    alpha_de = 0.8

    line_style_de = '--'
    line_style_gp = '-'

    lw_constr = 1
    t_range = [gp_error['time'][0], gp_error['time'][-1]]

    fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=set_size(textWidth, frac, (rows, cols), heightScale=h_scale))

    lines_1_gp = []
    lines_1_de = []
    lines_2_gp = []
    lines_2_de = []

    for i in range(len(header)):
        color = f'C{i}'
        if 'u' in header[i]:
            lines_2_gp.append(ax2.plot(gp_error['time'], gp_error[f'f{i+1}'], line_style_gp, color=color, alpha=alpha_gp)[0])
            lines_2_de.append(ax2.plot(de_error['time'], de_error[f'f{i+1}'], line_style_de, color=color, alpha=alpha_de)[0])
        else:
            lines_1_gp.append(ax1.plot(gp_error['time'], gp_error[f'f{i+1}'], line_style_gp, color=color, alpha=alpha_gp)[0])
            lines_1_de.append(ax1.plot(de_error['time'], de_error[f'f{i+1}'], line_style_de, color=color, alpha=alpha_de)[0])


    labels_1 = ['LODEGP $x_1$', 'LODEGP $x_2$', 'lin. model $x_1$', 'lin. model $x_2$',]
    labels_2 = ['LODEGP $u_1$', 'lin. model $u_1$']
    # lines_1_ref = (lines_1_ref[0], lines_1_ref[1])

    handles_1 = [lines_1_gp[0], lines_1_gp[1], lines_1_de[0], lines_1_de[1],]
    handles_2 = [lines_2_gp[0], lines_2_de[0]]
    columnspacing = 0.5
    handletextpad = 0.5
    handlelength = 1.5
    ax1.legend(handles_1, labels_1, handler_map={tuple: HandlerTuple(ndivide=None)}, fancybox=True, framealpha=0.5,ncol=2, columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength)#, loc='upper right',
    ax2.legend(handles_2, labels_2, handler_map={tuple: HandlerTuple(ndivide=None)}, fancybox=True, framealpha=0.5,ncol=2, columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength)#, loc='upper right',

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)#, wspace=-0.8 
    # ax1.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=3)#, ncol=stateN, columnspacing=columnspacing, handletextpad=handletextpad
    # ax2.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=2)
    ax2.set_xlabel(xLabel)
    ax1.set_ylabel(yLabel)
    ax2.set_ylabel('Absolute error ($\mathrm{m^3/s}$)')
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xticklabels([])
    ax1.set_xlim(t_range[0], t_range[1])
    ax2.set_xlim(t_range[0], t_range[1])
    # ax1.set_ylim(0.05-0.005, 0.255)
    # ax2.set_ylim(0-0.00001, 0.00015+0.00001)


    #ax2.yaxis.get_offset_text().set_position((-0.1, -0.5))
    #ax1.set_yticklabels([0.05,0.1,0.15,0.2,0.25])

    plt.plot()
    return fig

dflt_data_names = ['LODEGP', 'Training',  'lin. model']
dflt_xLabel = 'Time ($\mathrm{s})$'
dflt_yLabel = ['Fill level ($\mathrm{m}$)', 'Flow rate ($\mathrm{m^3/s}$)']
dflt_headers = ['x1', 'x2', 'u']

def plot_states(data, data_names=dflt_data_names, header=dflt_headers, xLabel=dflt_xLabel, yLabel=dflt_yLabel, uncertainty=None, title=None):
    init()
    rows = 2
    cols = 1
    frac = 1
    h_scale = 0.6

    alpha = [0.8, 0.8, 0.8]
    line_style = [ '-', '--' , '.']

    alpha_uncertainty = 0.1

    lw_constr = 1
    t_range = [data[0].time[0], data[0].time[-1]] #TODO

    fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=set_size(textWidth, frac, (rows, cols), heightScale=h_scale))

    lines_1 = [[] for j in range(len(data))]
    lines_2 = [[] for j in range(len(data))]

    for i in range(len(header)):
        color = f'C{i}'
        for j in range(len(data)):
            if 'u' in header[i]:
                lines_2[j].append(ax2.plot(data[j].time, data[j].y[:,i], line_style[j], color=color, alpha=alpha[j])[0])
                if data[j].uncertainty is not None:
                    ax2.fill_between(data[j].time, data[j].uncertainty['lower'][:,i], data[j].uncertainty['upper'][:,i], alpha=alpha_uncertainty, color=color)
            else:
                lines_1[j].append(ax1.plot(data[j].time, data[j].y[:,i], line_style[j], color=color, alpha=alpha[j])[0])
                if data[j].uncertainty is not None:
                    ax1.fill_between(data[j].time, data[j].uncertainty['lower'][:,i], data[j].uncertainty['upper'][:,i], alpha=alpha_uncertainty, color=color)

    labels_2 = [f'{data_names[j]} {header[-1]}' for j in range(len(data))]
    labels_1 = []
    handles_1 = []
    handles_2 = []
    for j in range(len(data)):
        for i in range(len(header)-1):
            labels_1.append(f'{data_names[j]} {header[i]}')
            handles_1.append(lines_1[j][i])
        handles_2.append(lines_2[j][0])

    columnspacing = 0.5
    handletextpad = 0.5
    handlelength = 1.5
    ax1.legend(handles_1, labels_1, handler_map={tuple: HandlerTuple(ndivide=None)},  fancybox=True, framealpha=0.5,ncol=2, columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength) #loc='upper right',
    ax2.legend(handles_2, labels_2, handler_map={tuple: HandlerTuple(ndivide=None)}, fancybox=True, framealpha=0.5,ncol=2, columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength) #loc='upper right',

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)#, wspace=-0.8 
    if title is not None:
        ax1.set_title(title)
    # ax1.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=3)#, ncol=stateN, columnspacing=columnspacing, handletextpad=handletextpad
    # ax2.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=2)
    ax2.set_xlabel(xLabel)
    ax1.set_ylabel(yLabel[0])
    ax2.set_ylabel(yLabel[1])
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xticklabels([])
    ax1.set_xlim(t_range[0], t_range[1])
    ax2.set_xlim(t_range[0], t_range[1])
    # ax1.set_ylim(0.05-0.005, 0.255)
    # ax2.set_ylim(0-0.00001, 0.00015+0.00001)

    #ax2.yaxis.get_offset_text().set_position((-0.1, -0.5))
    #ax1.se
    return fig


def save_plot_to_pdf(fig, fileName:str):
    fig.savefig(imgFolder + fileName + '.pdf', format=fileFormat, bbox_inches='tight')
