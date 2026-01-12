import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.gridspec import GridSpec

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from utils.metrics_tools import ASE
LabelSize = 10
TitleSize = 15
LegSize = 10


def MakeFrac(series, min_d=1):
    # sorted by trace index
    Sseries = series[np.argsort(series[:, 1])]
    # split non-continue series
    NewSeriesList = []
    start = 0
    for i in range(1, Sseries.shape[0]):
        if Sseries[i, 1] - Sseries[i-1, 1] > min_d or i == Sseries.shape[0]-1:
            NewSeriesList.append(Sseries[start:(i+1), :])
            start = i
    return NewSeriesList


def insert_zeros(trace, tt=None):
    """Insert zero locations in data trace and tt vector based on linear fit"""

    if tt is None:
        tt = np.arange(len(trace))

    # Find zeros
    zc_idx = np.where(np.diff(np.signbit(trace)))[0]
    x1 = tt[zc_idx]
    x2 = tt[zc_idx + 1]
    y1 = trace[zc_idx]
    y2 = trace[zc_idx + 1]
    a = (y2 - y1) / (x2 - x1)
    tt_zero = x1 - y1 / a

    # split tt and trace
    tt_split = np.split(tt, zc_idx + 1)
    trace_split = np.split(trace, zc_idx + 1)
    tt_zi = tt_split[0]
    trace_zi = trace_split[0]

    # insert zeros in tt and trace
    for i in range(len(tt_zero)):
        tt_zi = np.hstack(
            (tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
        trace_zi = np.hstack(
            (trace_zi, np.zeros(1), trace_split[i + 1]))

    return trace_zi, tt_zi


def wiggle_plot(gather, curve_dict=None, seg_map=None, curve_dict_m=None, save_path=None, size=(10, 10), title = None, dpi=500, plot_ase=False):
    # ----- trace-wise normalization ------
    gather_ori = gather.copy()
    gather = gather / (np.max(np.abs(gather), axis=0)+1e-10) * 0.6
    # ----- plot part --------
    tVec = np.arange(0, int(gather.shape[0]))
    oVec = np.arange(0, int(gather.shape[1]), 1)
    fig = plt.figure(figsize=size)
    axs = GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=(4, 1)) 
    ax = fig.add_subplot(axs[0])
    # https://coolors.co/palette/264653-2a9d8f-e9c46a-f4a261-e76f51
    C_list = ['#'+col for col in '264653-2a9d8f-e9c46a-f4a261-e76f51'.split('-')]
    for i in range(gather.shape[1]):
        trace_pad, tt_pad = insert_zeros(gather[:, i], tVec)
        ax.plot(oVec[i] + trace_pad, tt_pad, color='black', linewidth=0.1)
        ax.fill_betweenx(tt_pad, oVec[i], trace_pad + oVec[i],
                         where=trace_pad >= 0, facecolor='black')
    c_count = 0
    # plot first break times on the wiggle plot
    if curve_dict is not None:
        # sort the curve
        for curve_name in curve_dict.keys():
            curve_array = np.array(curve_dict[curve_name])
            curve_array = curve_array[:, [1, 0]]
            # split non-continue series
            SeriesList = MakeFrac(curve_array, min_d=1)
            for k, ser in enumerate(SeriesList):
                # ax.scatter(ser[:, 1], ser[:, 0], marker='o', s=0.5, c=C_list[c_count%len(C_list)], alpha=0.5)
                ax.plot(ser[:, 1]-1, ser[:, 0]-1, c=C_list[c_count%len(C_list)], alpha=0.7)
            c_count+=1
            
    if curve_dict_m is not None:
        # sort the curve
        for curve_name in curve_dict_m.keys():
            curve_array = np.array(curve_dict_m[curve_name])
            curve_array = curve_array[:, [1, 0]]
            # split non-continue series
            SeriesList = MakeFrac(curve_array, min_d=1)
            for k, ser in enumerate(SeriesList):
                ax.plot(ser[:, 1]-1, ser[:, 0]-1, c='blue', alpha=1, linewidth=0.4)
             
    if seg_map is not None:
        y, x = np.where(seg_map>0.5)
        point_array = np.array([x, y]).T
        ax.scatter(point_array[:, 0], point_array[:, 1], s=3, marker='>', c='red')
        
    ax.set_xlim(min(oVec)-1, max(oVec)+1)
    ax.set_ylim(0, len(tVec))
    ax.set_xlabel('Trace Index', fontsize=LabelSize)
    ax.set_ylabel('Depth', fontsize=LabelSize)
    ax.invert_yaxis()
    
    if plot_ase:
        if curve_dict is not None:
            ax2 = fig.add_subplot(axs[1])
            ase_list = ASE(gather_ori, curve_dict.values())
            y_list = [np.mean(curve[:, 1]) for curve in curve_dict.values()]
            ax2.stem(y_list, ase_list, linefmt='grey', markerfmt='D', bottom=0.2, label='Auto', orientation='horizontal')
            ax2.set_xlabel('ASE')
            ax2.set_ylim(0, len(tVec))
            ax2.invert_yaxis()
        if curve_dict_m is not None:
            ase_list = ASE(gather_ori, curve_dict_m.values())
            y_list = [np.mean(curve[:, 1]) for curve in curve_dict_m.values()]
            ax2.stem(y_list, ase_list, bottom=0.1, label='Manual', orientation='horizontal')
                    
    if title is not None:
        plt.title(title, fontsize=TitleSize)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close('all')
        

def heatmap_plot(gather, curve_dict=None, seg_map=None, save_path=None, 
               size=(10, 10), cmap='binary', title = None, dpi=500):
    # gather = gather / (np.max(np.abs(gather), axis=0)+1e-10) * 0.7
    fig, ax = plt.subplots(figsize=size)
    cax = ax.imshow(gather, cmap=cmap, aspect='auto')
    fig.colorbar(cax, ax=ax, pad=0.02, fraction=0.02)
    ax.set_xlabel('Trace Index', fontsize=LabelSize)
    ax.set_ylabel('Depth', fontsize=LabelSize)
    C_list = ['#'+col for col in '264653-2a9d8f-e9c46a-f4a261-e76f51'.split('-')]
    c_count = 0
    # plot first break times on the wiggle plot
    if curve_dict is not None:
        for curve_name in curve_dict.keys():
            curve_array = np.array(curve_dict[curve_name])
            curve_array = curve_array[:, [1, 0]]
            # split non-continue series
            SeriesList = MakeFrac(curve_array, min_d=1)
            for k, ser in enumerate(SeriesList):
                ax.plot(ser[:, 1]-1, ser[:, 0]-1, linewidth=0.3, c=C_list[c_count%len(C_list)])
            c_count+=1
    if seg_map is not None:
        y, x = np.where(seg_map>0.5)
        point_array = np.array([x, y]).T
        ax.scatter(point_array[:, 0], point_array[:, 1], s=3, marker='>', c='red')
        
    if title is not None:
        plt.title(title, fontsize=TitleSize)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close('all')
        

from scipy.stats import gaussian_kde
def compare_metrics(ase_dict, save_path=None, title=None, size=(6, 4), dpi=500):
    fig = plt.figure(figsize=size, dpi=200)
    c_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    ax = fig.add_subplot(111)
    ase_all_value = []
    for k, name in enumerate(ase_dict):
        ase = np.array(ase_dict[name]).reshape(-1)
        if len(ase) > 0:
            v = np.linspace(ase.min()-0.01, ase.max()+0.01, 100)
            ase_all_value += list(ase)
            kde_opt = gaussian_kde(ase)
            ase_kde = kde_opt(v)
            # ase_kde = ase_kde / (ase_kde.max() + 1e-10)
            ax.fill_between(v, 0, ase_kde, alpha=0.5, color=c_list[k], label=name)
    ax.set_xlabel(title, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close('all')
    
    
def plot_log(log_dict, save_root):
    for k, (name, log_list) in enumerate(log_dict.items()):
        fig = plt.figure(figsize=(6, 4), dpi=200)
        ax = fig.add_subplot(111)
        ax.plot((np.arange(len(log_list))+1).astype(np.int32), log_list, 'o', ls='--', color='gray',
                linewidth=1, markersize=2, markeredgewidth=0.1, markerfacecolor='tab:blue', markeredgecolor='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name)
        plt.savefig(os.path.join(save_root, 'log_%s.png' % name), bbox_inches='tight', pad_inches=0.1)
        plt.close('all')


def visual_cluster(gth, curve_dict, curve_info, labels, save_path, cmap='binary'):
    fig, ax = plt.subplots(figsize=(3, 30))
    cax = ax.imshow(gth, cmap=cmap, aspect='auto')
    fig.colorbar(cax, ax=ax, pad=0.02, fraction=0.02)
    ax.set_xlabel('Trace Index', fontsize=LabelSize)
    ax.set_ylabel('Depth', fontsize=LabelSize)
    C_list = ['#'+col for col in '264653-2a9d8f-e9c46a-f4a261-e76f51'.split('-')]
    c_count = 0
    
    for id, curve_name in enumerate(curve_dict.keys()):
        curve_array = np.array(curve_dict[curve_name])
        curve_array = curve_array[:, [1, 0]]
        # split non-continue series
        SeriesList = MakeFrac(curve_array, min_d=1)
        for k, ser in enumerate(SeriesList):
            ax.scatter(ser[:, 1]-1, ser[:, 0]-1, marker='o', s=1, c=C_list[c_count%len(C_list)], alpha=1)
        c_count+=1
        x, y, _, _ = curve_info[curve_name]
        ax.text(x-1, y-1, labels[id]+1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')


def plot_multi_fig(map_list, name_list, col_list, lab_col_list, cmap_list, bar_name_list, pick_list=None, save_path=None, fig_size=(4, 8), col_range=None):
    h=0.05
    # assign plot group
    names = np.unique(bar_name_list)
    group_ind = [np.where(np.array(bar_name_list) == name)[0] for name in names]
    v_range = np.zeros((len(map_list), 2))
    for group_list in group_ind:
        map_list_array = np.array([map_list[i] for i in group_list])
        v_max_k = np.max(np.abs(map_list_array))
        v_min_k = -v_max_k
        # v_min_k, v_max_k = np.min(map_list_array), np.max(map_list_array)
        if col_range is not None:
            v_min_k, v_max_k = col_range
        v_range[group_list, 0] = v_min_k
        v_range[group_list, 1] = v_max_k
    colorbar_plot = [group_list[-1] for group_list in group_ind]
    bar_start_position = [0.91, 0.7, 0.008, 0.2]
    
    # plot subfigs
    fig, axs = plt.subplots(1, len(map_list), figsize=fig_size, gridspec_kw={'wspace': 0.01, 'hspace': 0})
    for i, ax in enumerate(axs):
        cbar_k = ax.imshow(map_list[i], cmap=cmap_list[i], aspect='auto', vmin=v_range[i, 0], vmax=v_range[i, 1])
        if i > 0:
            ax.axis('off')
        else:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xlabel('Trace Index')
            ax.set_ylabel('Depth Samples')
        if i in colorbar_plot:
            cbar_ax = fig.add_axes(bar_start_position)
            cbar = plt.colorbar(cbar_k, fraction=0.05, cax=cbar_ax, label=bar_name_list[i])
            cbar.mappable.set_clim(v_range[i, 0], v_range[i, 1])
            bar_start_position[1] -= 0.25
            
        # bottom title
        rect = plt.Rectangle((0.01, 1.001), 1-0.01, h, color=col_list[i], transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.5, 1+h/2, name_list[i], ha='center', va='center', transform=ax.transAxes, fontsize=5, color=lab_col_list[i])
        # plot curvatures
        C_list = ['#'+col for col in '264653-2a9d8f-e9c46a-f4a261-e76f51'.split('-')]
        c_count = 0
        if pick_list is not None: 
            pick_dict = pick_list[i]
            for k, (name, curves) in enumerate(pick_dict.items()):
                for j, curve in enumerate(curves.values()):
                    curve_array = np.array(curve)
                    curve_array = curve_array[:, [1, 0]]
                    # split non-continue series
                    SeriesList = MakeFrac(curve_array, min_d=1)
                    if name  == 'manual':
                        color = 'red'
                        lab = 'manual'
                    else:
                        color = C_list[c_count%len(C_list)]
                        lab = None
                    for k, ser in enumerate(SeriesList):
                        ax.plot(ser[:, 1]-1, ser[:, 0]-1, linewidth=0.5, c=color, label=lab, alpha=0.7)
                    c_count+=1
                    
        ax.set_ylim(map_list[i].shape[0], 100)
        
        if i == 0 and pick_list is not None:
            # make sure the unique legend
            handles, labels = ax.get_legend_handles_labels()
            unique_handles, unique_labels = [], []
            seen_labels = set()
            for handle, label in zip(handles, labels):
                if label not in seen_labels:
                    unique_handles.append(handle)
                    unique_labels.append(label)
                    seen_labels.add(label)
            if lab is not None:
                ax.legend(unique_handles, unique_labels, fontsize=8, loc='lower center', ncol=2, bbox_to_anchor=(1.5, -0.08))
            
    if save_path is not None:  
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01)
    else:
        plt.show()
    plt.close('all')
    

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

def plot_zoom(gather, curve_dict=None, save_path=None, cmap='binary', size=(2, 5), plot_axis=True, clu_lab=None, curve_m_dict=None, assign_col=None):
    LabelSize = 10
    fig = plt.figure(figsize=size)  
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)
    ax = plt.subplot(gs[0])

    def plot_single(ax_k, zoom=1):
        cax = ax_k.imshow(gather, cmap=cmap, aspect='auto', origin='upper')
        C_list = ['#'+col for col in 'ef476f-ffd166-06d6a0-118ab2-073b4c'.split('-')]
        c_count = 0
        if curve_dict is not None:
            for i, curve_name in enumerate(curve_dict.keys()):
                curve_array = np.array(curve_dict[curve_name])
                curve_array = curve_array[:, [1, 0]]
                # split non-continue series
                SeriesList = MakeFrac(curve_array, min_d=1)
                color = assign_col if assign_col is not None else C_list[c_count%len(C_list)]
                for k, ser in enumerate(SeriesList):
                    label='cascade' if i==0 and k == 0 and assign_col is not None else None
                    ax_k.plot(ser[:, 1]-1, ser[:, 0]-1, linewidth=0.6*zoom, c=color, alpha=0.7, label=label)
                if clu_lab is not None:
                    if i<len(list(curve_dict.keys()))-1:
                        if clu_lab[i+1]!=clu_lab[i]:
                            c_count+=1
                else:
                    c_count+=1   
        if curve_m_dict is not None:
            for i, curve_name in enumerate(curve_m_dict.keys()):
                curve_array = np.array(curve_m_dict[curve_name])
                curve_array = curve_array[:, [1, 0]]
                # split non-continue series
                SeriesList = MakeFrac(curve_array, min_d=1)
                for k, ser in enumerate(SeriesList):
                    label='manual' if i==0 and k == 0 else None
                    # ax_k.scatter(ser[:, 1]-1, ser[:, 0]-1, s=0.1*zoom, c='red', alpha=1, linewidths=None, edgecolors=None, label=label)
                    ax_k.plot(ser[:, 1]-1, ser[:, 0]-1, linewidth=0.6*zoom, c='red', alpha=0.7, label=label)
            ax.legend(fontsize=6, loc='lower center', ncol=1, bbox_to_anchor=(1.6, -0.1))
        return cax
    
    cax = plot_single(ax)
    ax.set_xlim(0, gather.shape[1])
    ax.set_ylim(800, 100)
    # fig.colorbar(cax, ax=ax, pad=0.02, fraction=0.02)
    if plot_axis:
        ax.set_ylabel('Depth Samples', fontsize=LabelSize)
    else:
        ax.set_yticks([])
    ax.set_xlabel('Trace Index', fontsize=LabelSize)
    gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], hspace=0.3)

    ax_1 = plt.subplot(gs_right[0])
    ax_2 = plt.subplot(gs_right[1])
    ax_3 = plt.subplot(gs_right[2])

    x1, x2 = 5, 45
    for k, (color, y) in enumerate(zip(['#fcbf49', '#d62828', '#f77f00'], [200, 400, 650])):
        h = 120
        ax_k = eval('ax_%s'%(k+1))
        plot_single(ax_k, 2)
        ax_k.set_xlim(x1, x2)
        ax_k.set_ylim(min(y+h, gather.shape[0])-5, y)
        ax_k.tick_params(labelleft=False, labelbottom=False, color=color)
        # ax_k.yaxis.get_major_locator().set_params(nbins=0)
        # ax_k.xaxis.get_major_locator().set_params(nbins=0)
        ax_k.set_yticks([])
        ax_k.set_xticks([])
        ax_k.spines['top'].set_color(color)     
        ax_k.spines['bottom'].set_color(color)  
        ax_k.spines['left'].set_color(color)   
        ax_k.spines['right'].set_color(color) 
        rect = patches.Rectangle((x1, y), x2-x1, min(h, gather.shape[0]-y)-5, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01)
    plt.close()