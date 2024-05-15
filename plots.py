import matplotlib.pyplot as plt
from potentials import ArbitraryPolynomialPotential
from trap_model import TrapModel
import sys
import matplotlib.pyplot as plt
import numpy as np
import os

#Can be easily implemented into Interp_Sampled_Potential_Lines_class by replacing the vertex args with self
def plot_mode_matrix(vertex=Voltage_vertex, split_n=None):
    fig,ax=plt.subplots()
    modes=(vertex.calc_radial_frequencies()[1])
    cax=ax.matshow(modes, cmap='RdBu', vmin=-np.max(np.abs(modes)), vmax=np.max(np.abs(modes)),)
    ax.set_title('Normalized Radial Mode Participation')
    ax.xaxis.set_ticks_position('bottom')
    if split_n is not None: ax.plot(-.5,split_n-.5, color='black', marker='>', linestyle='none', zorder=100, 
                                    clip_on=False, markersize=10)
    ax.set_xlabel('Mode #')
    ax.set_ylabel('Ion #')

    cbar=plt.colorbar(cax)
    cbar.set_label('Participation')
    return ax

def plot_modes(vertex=Voltage_vertex, include_axial=False):
    fig, ax=plt.subplots(2,1)
    modes=(vertex.calc_radial_frequencies()[0])
    modes_axial=[vertex.calc_axial_frequencies()[0]]
    fig.set_figheight(5)
    fig.set_figwidth(15)
    ax[0].set_title('Radial Mode Frequencies')
    ax[1].set_title('Axial Mode Frequencies')
    ax[0].vlines(modes, ymin=0, ymax=.8*np.ones(len(modes)), colors=['blue']*len(modes))
    ax[1].vlines(modes_axial[0], ymin=0, ymax=.8*np.ones(len(modes_axial)), colors=['red']*len(modes_axial))
    ax[0].vlines(3, ymin=0, ymax=1, alpha=0)
    ax[1].vlines(.4, ymin=0, ymax=1, alpha=0)
    ax[0].set_xticks(np.linspace(2.5,3, 15))
    ax[0].set_xticklabels(['{:.3}'.format(i) for i in np.linspace(2.5,3, 15)])
    ax[1].set_xticks(np.linspace(.17,.5, 15))
    ax[1].set_xticklabels(['{:.3}'.format(i) for i in np.linspace(.17,.4, 15)])
    for axes in ax:
        axes.set_yticks([])
        axes.set_ylabel(None)
        axes.grid(True)
        axes.margins(y=0)
        axes.tick_params(left=False)
        
    ax[1].set_xlabel(r'Mhz/$2\pi$')
    plt.tight_layout()
    return ax

def plot_potential(vertex=Voltage_vertex, x_range=np.ndarray, include_ez=False, include_ions=False, include_raman=False, is_plot_contraints=True, N_pnts=401):
    fig, ax=plt.subplots(1,2)
    xs = np.linspace(x_range[0], x_range[1], N_pnts)
    vertex.spline_V = vertex.HOA.calc_V_spline(vertex.Voltages)

    if include_raman:
        Raman_distance = 4.4
        Original_Raman_positions = np.linspace(-Raman_distance * int(vertex.N_ions / 2),
                                                Raman_distance * int(vertex.N_ions / 2), vertex.N_ions)
        ax[1].plot(Original_Raman_positions, np.zeros_like(Original_Raman_positions), marker="x", linestyle="none",
                    linewidth=0.5, color="palegreen", markersize=1.5)
        
    ax[1].plot(xs * 1e3, 1e3 * vertex.spline_V(xs), marker="", linestyle="solid", linewidth=2, color="indigo")

    if is_plot_contraints:
        ax[1].plot(np.array(vertex.x_maxima) * 1e3, 1e3 * vertex.spline_V(vertex.x_maxima), marker="d", linestyle="none",
                    linewidth=1, color="saddlebrown", markersize=3.5)
        ax[1].plot(np.array(vertex.x_minima) * 1e3, 1e3 * vertex.spline_V(vertex.x_minima), marker="d", linestyle="none",
                    linewidth=1, color="salmon", markersize=3.5)
        ax[1].plot(np.array(vertex.x_saddle) * 1e3, 1e3 * vertex.spline_V(vertex.x_saddle), marker="s", linestyle="none",
                    linewidth=1, color="darkgreen", markersize=2)
        
    if include_ez:
        ax[1].twinx()
        ax[1].plot(xs*1e3, 1e3*vertex.spline_Ez(xs), marker="", linestyle="solid", linewideth=2, color='red')
    if include_ions: 
        ax[0].twinx(ax[1])
        ax[0].scatter(model.pos*1e6, np.zeros(model.pos.__len__()))
        ax[0].set_tick_params(left=False)
        ax[0].set_yticks([])
        ax[0].set_ylabel('Ion Positions')
            

    ax[1].set_xlabel(r'x ($\mu$m)')
    ax[1].set_ylabel(r'$\Phi(x)$ (meV)')


    return ax

def plot_ion_pos_opt_cost_function(graph=Voltage_graph, vertex_ind=int, show=False):
    fig,ax=plt.subplots()
    costs=graph.callback_log_list[vertex_ind]
    ax.plot(costs)
    ax.grid(True)
    ax.set_ylabel('Cost')
    ax.set_xlabel('Iterations')
    ax.set_title('Cost Function V. Iterations')
    if not show: plt.show(False)
    return ax

def plot_opt_cost_function(graph=Voltage_graph, show=False):
    fig,ax=plt.subplots()
    costs_fin=[i[-1] for i in graph.callback_log_list]
    ax.set_ylabel('Cost at Final Iteration')
    ax.set_xlabel('Vertex Index')
    ax.set_title('Cost Function V. Vertexes')
    if not show: plt.show(False)
    return ax

def plot_final_cost_v_vertex(graph=Voltage_graph, show=False):
    fig,ax=plt.subplots()
    costs_fin=[i[-1] for i in graph.callback_log_list]
    ax.plot(costs_fin)
    ax.set_ylabel('Cost at Final Iteration')
    ax.set_xlabel('Vertex Index')
    plt.grid(True)
    ax.set_title('Cost Function V. Vertexes')
    if not show: plt.show(False)
    return ax

