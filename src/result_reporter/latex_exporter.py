
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
        color = f'C{i}'
        if 'u' in header[i]:
            ax2.plot(training_data['time'], training_data[f'f{i+1}'], '.', label=f'{header[i]} train', color=color, alpha=alpha_train)
            ax2.plot(sim_data['time'], sim_data[f'f{i+1}'], label=header[i], color=color, alpha=alpha_est)
            if reference_data and len(reference_data) > 0: 
                ax2.plot(reference_data['time'], reference_data[f'f{i+1}'], '--', label=f'{header[i]} ref', color=color, alpha=alpha_est)
            
        else:
            ax1.plot(training_data['time'], training_data[f'f{i+1}'], '.',label=f'{header[i]} train',  color=color, alpha=alpha_train)
            ax1.plot(sim_data['time'], sim_data[f'f{i+1}'], label=header[i], color=color, alpha=alpha_est)
            if reference_data and len(reference_data) > 0: 
                ax1.plot(reference_data['time'], reference_data[f'f{i+1}'], '--', label=f'{header[i]} ref', color=color, alpha=alpha_est)
        

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

def create_mpc_plot(training_data, sim_data, header:List[str], xLabel='Time ($\mathrm{s})$', yLabel='Water Level ($\mathrm{m}$)', reference_data=None, x_e=None, title=None):
    width = '8.4 cm'

    textWidth_cm = 8.4

    init()
    rows = 1
    cols = 1
    scale = 1.5
    frac = 1

    alpha_train = 1
    alpha_est = 1
    alpha_ref = 0.7

    fig, ax1 = plt.subplots(rows, cols, figsize=set_size(scale*textWidth_cm * CM_to_PT, frac, (rows, cols)))
    ax2 = ax1.twinx()

    if x_e is not None:
        equilibrium = np.array([x_e, x_e])
        t_range = [training_data['time'][0], reference_data['time'][-1]]


    for i in range(len(header)):
        color = f'C{i}'
        if 'u' in header[i]:
            #ax2.plot(training_data['time'], training_data[f'f{i+1}'], 'o', label=f'$u_1$ setpoint', color=color, alpha=alpha_train)
            #ax2.plot(sim_data['time'], sim_data[f'f{i+1}'], label=header[i], color=color, alpha=alpha_est)
            ax2.plot(reference_data['time'], reference_data[f'f{i+1}'], label='$u_1$', color=color, alpha=alpha_est)
            ax2.plot(t_range, equilibrium[:,i], '--', label='$u_1$ Reference',color=color, alpha=alpha_ref)
            
        else:
            #ax1.plot(training_data['time'], training_data[f'f{i+1}'], 'o',label=f'$x_{i}$ setpoint',  color=color, alpha=alpha_train)
            #ax1.plot(sim_data['time'], sim_data[f'f{i+1}'], label=header[i], color=color, alpha=alpha_est)
            ax1.plot(reference_data['time'], reference_data[f'f{i+1}'], label=f'$x_{i}$', color=color, alpha=alpha_est) #'--', 
            ax1.plot(t_range, equilibrium[:,i], '--', label=f'$x_{i}$ Reference',color=color, alpha=alpha_ref)
        
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    # ax1.legend()
    # ax2.legend()
    ax1.set_xlabel(xLabel)
    ax1.set_ylabel(yLabel)
    ax2.set_ylabel('input flow rate ($\mathrm{m^3/s}$)')
    ax1.grid(True)
    if title:
        ax1.set_title(title)

    fig.tight_layout()
    return fig

def create_mpc_plot_2(training_data, sim_data, header:List[str], xLabel='Time ($\mathrm{s})$', yLabel='Water Level ($\mathrm{m}$)', reference_data=None, x_e=None,  close_constraint=False):
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

    fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=set_size(textWidth_cm * CM_to_PT, frac, (rows, cols), heightScale=h_scale))
    

    if x_e is not None:
        equilibrium = np.array([x_e, x_e])
        t_range = [reference_data['time'][0], reference_data['time'][-1]]

        lower = [0, 0, 0]
        upper = [0.6, 0.6,2e-4]
        lower_constraint = np.array([lower, lower])
        upper_constraint = np.array([upper, upper] )

        ref_line = '--'
        constraint_line = ':'

    for i in range(len(header)):
        color = f'C{i}'
        gray = 'C7'
        if 'u' in header[i]:
            #ax2.plot(training_data['time'], training_data[f'f{i+1}'], 'o', label=f'$u_1$ setpoint', color=color, alpha=alpha_train)
            #ax2.plot(sim_data['time'], sim_data[f'f{i+1}'], label=header[i], color=color, alpha=alpha_est)
            ax2.plot(reference_data['time'], reference_data[f'f{i+1}'], label='$u_1$', color=color, alpha=alpha_est)
            ax2.plot(t_range, equilibrium[:,i], ref_line,color=color, alpha=alpha_ref) #, label='$u_1$ Reference'
            
            if close_constraint:
                ax2.plot(t_range, equilibrium[:,i]*0.9, constraint_line, label='Constraints', color=gray, alpha=alpha_ref)
                ax2.plot(t_range, equilibrium[:,i]*1.1, constraint_line,  color=gray, alpha=alpha_ref)
            else:
                ax2.plot(t_range, lower_constraint[:,i], constraint_line, label='Constraints', color=gray, alpha=alpha_ref)
                #ax2.plot(t_range, upper_constraint[:,i], '--',  color=gray, alpha=alpha_ref)

            
            
        else:
            #ax1.plot(training_data['time'], training_data[f'f{i+1}'], 'o',label=f'$x_{i}$ setpoint',  color=color, alpha=alpha_train)
            #ax1.plot(sim_data['time'], sim_data[f'f{i+1}'], label=header[i], color=color, alpha=alpha_est)
            ax1.plot(reference_data['time'], reference_data[f'f{i+1}'], label=f'$x_{i+1}$', color=color, alpha=alpha_est) #'--', 
            

            if i == 1:
                line1 = ax1.plot(t_range, equilibrium[:,i], ref_line, label=f'$Reference', color=color, alpha=alpha_ref)
            else:
                line0 = ax1.plot(t_range, equilibrium[:,i], ref_line, color=color, alpha=alpha_ref)#label=f'$x_{i+1}$ Reference',

            if close_constraint:
                if i == 1:
                    ax1.plot(t_range, equilibrium[:,i]*0.9, constraint_line, label='Constraints', color=gray, alpha=alpha_ref)
                else:
                    ax1.plot(t_range, equilibrium[:,i]*0.9, constraint_line, color=gray, alpha=alpha_ref)
                ax1.plot(t_range, equilibrium[:,i]*1.1, constraint_line,  color=gray, alpha=alpha_ref)
            # else:
            #     if i == 0:
            #         ax1.plot(t_range, lower_constraint[:,i], '--', label='Constraints', color=gray, alpha=alpha_ref)
            #     else:
            #         ax1.plot(t_range, lower_constraint[:,i], '--', color=gray, alpha=alpha_ref)
            #     ax1.plot(t_range, upper_constraint[:,i], '--', color=gray, alpha=alpha_ref)


    from matplotlib.legend_handler import HandlerTuple
    handles = [(line0, line1)]
    labels = ['Reference']
    ax1.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)})

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)#, wspace=-0.8 
    ax1.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=3)#, ncol=stateN, columnspacing=columnspacing, handletextpad=handletextpad
    ax2.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=2)
    ax2.set_xlabel(xLabel)
    ax1.set_ylabel(yLabel)
    ax2.set_ylabel('Flow rate ($\mathrm{m^3/s}$)')
    ax1.grid(True)
    ax2.grid(True)
    #ax2.yaxis.get_offset_text().set_position((-0.1, -0.5))
    ax1.set_xticklabels([])
    #ax1.set_yticklabels([0.05,0.1,0.15,0.2,0.25])
    ax1.set_xlim(t_range[0], t_range[1])
    ax2.set_xlim(t_range[0], t_range[1])

    ax1.set_ylim(0.05-0.005, 0.255)
    ax2.set_ylim(0-0.00001, 0.00015+0.00001)

    
    return fig



def create_mpc_plot_3(training_data, sim_data, header:List[str], xLabel='Time ($\mathrm{s})$', yLabel='Water Level ($\mathrm{m}$)', reference_data=None, x_e=None,  close_constraint=False):
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


def save_plot_to_pdf(fig, fileName:str):
    fig.savefig(imgFolder + fileName + '.pdf', format=fileFormat, bbox_inches='tight')
