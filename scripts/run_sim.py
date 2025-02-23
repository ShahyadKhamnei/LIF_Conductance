import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import welch
import seaborn as sns

from src.model import intensity
from src.theory import lif_rate_homog, fixed_pt_iter_propagators_1pop, fixed_pt_iter_propagators_1pop_true
from src.sim import sim_lif_perturbation, sim_lif_pop, create_spike_train

fontsize = 10
labelsize = 9

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['k']+colors

root_dir = '/Users/gkocker/Documents/projects/lif_stochastic'
root_dir = '/Users/micha/School/Neuro/LIF_Conductance'
results_dir = os.path.join(root_dir, 'results')


def plot_fig_single_pop_weakly_coupled(savedir=results_dir, Jmax=10, Emax=2):

    # fig, ax = plt.subplots(2, 2, figsize=(3.4, 3.7))
    fig = plt.figure()

    ### simulation examples of bistability

    N = 100
    connection_prob = 0.5

    tstop = 20
    dt = .01
    tplot = np.arange(0, tstop, dt)
    E = 0
    perturb_amp = 2
    perturb_len = 2

    Nt = int(tstop/dt)
    E_plot = np.zeros(Nt,) + E
    t_start_perturb1 = Nt//4
    t_end_perturb1 = t_start_perturb1 + int(perturb_len / dt)

    t_start_perturb2 = 3*Nt//4
    t_end_perturb2 = t_start_perturb2 + int(perturb_len / dt)

    E_plot[t_start_perturb1:t_end_perturb1] += perturb_amp
    E_plot[t_start_perturb2:t_end_perturb2] -= perturb_amp

    Eshift = 15
    Escale = 5
    E_plot = Escale*E_plot + N + Eshift

    E_l = 0.5
    E_s = 3
    g = 4
    J = np.random.binomial(n=1, p=connection_prob, size=(N,N)) * g / connection_prob / N
    _, spktimes = sim_lif_perturbation(J=J, E_l=E_l, E_s=E_s, tstop=tstop, dt=dt, perturb_len=perturb_len, perturb_amp=perturb_amp)

    plt.plot(spktimes[:, 0], spktimes[:, 1], 'k|', markersize=1.5)

    raster_yticks = (0, N)
    raster_yticklabels = (0, N)

    plt.plot(tplot, E_plot, 'k')
    plt.title(f'J={g}, E_leak={E_l}, E_syn={E_s}')
    plt.xlim((0, tstop))
    plt.ylim((0, N+Eshift+3.5*Escale))
    plt.yticks(raster_yticks)
    # plt.yticklabels(raster_yticklabels)
    plt.xlabel('Time (ms/{})'.format(r'$\tau$'), fontsize=fontsize)
    # plt.set_ylabel('Neuron', fontsize=fontsize)

    fig.savefig(os.path.join(savedir, f'sim_perturb_J{g}_El{E_l}_Es{E_s}.png'), dpi=600)


def plot_fig_single_pop_bifurcation(savefile=os.path.join(results_dir, 'fig_homog_bif.pdf'), Jbounds=(.5, 5.5), Emax=2, eps=1e-11):    

    '''
    reproduce figure 2e, f
    '''

    fig, ax = plt.subplots(2, 1, figsize=(2, 3.7))

    E = .5
    connection_prob = 0.5
    N = 100
    tstop = 500
    tstim = 10
    Estim = 10

    Jmin, Jmax = Jbounds
    Jrange = np.arange(Jmin, Jmax, .5)

    r_sim = []
    r_sim_stim = []

    for g in Jrange:
        J = np.random.binomial(n=1, p=connection_prob, size=(N,N)) * g / connection_prob / N

        _, spktimes = sim_lif_pop(J=J, E=E, tstop=tstop, tstim=0, Estim=0)

        r_sim.append(len(spktimes) / N / tstop)
        
        _, spktimes = sim_lif_pop(J=J, E=E, tstop=tstop, tstim=tstim, Estim=Estim)
        
        spktimes = spktimes[spktimes[:, 0] > 2*tstim, :]
        if len(spktimes) > 0:
            r_sim_stim.append(len(spktimes) / N / (tstop - 2*tstim))
        else:
            r_sim_stim.append(np.nan)


    if E >= 1:
        Jrange_th = np.arange(Jmin, Jmax, .005)
    else:
        Jrange_th1 = np.linspace(Jmin, 2*(1+np.sqrt(1-E))-.01, 5)
        Jrange_th2 = np.arange(2*(1+np.sqrt(1-E))-.01, Jmax, .005)
        Jrange_th = np.concatenate((Jrange_th1, Jrange_th2))

    r_th = [lif_rate_homog(g, E) for g in Jrange_th]

    r_mft_low = 0 * Jrange_th
    v_mft_high = 0 * Jrange_th
    r_mft_high = v_mft_high.copy()

    Jbif = np.where(Jrange_th > 2*(1+np.sqrt(1-E)))[0][0]
    v_mft_high[Jbif:] = (Jrange_th[Jbif:] + np.sqrt(Jrange_th[Jbif:]**2 + 4*(E - Jrange_th[Jbif:]))) / 2
    r_mft_high = intensity(v_mft_high)

    ax[0].plot(Jrange, r_sim, 'ko', alpha=0.5)
    ax[0].plot(Jrange, r_sim_stim, 'ko', alpha=0.5)

    ax[0].plot(Jrange_th, r_mft_low, 'k', linewidth=2)
    ax[0].plot(Jrange_th, r_mft_high, 'k--', linewidth=2, label='1st ord.')
    ax[0].plot(Jrange_th, r_th, 'k', linewidth=2, label='exact')

    ax[0].legend(loc=0, frameon=False, fontsize=fontsize)
    ax[0].set_xlabel('J', fontsize=fontsize)
    ax[0].set_ylabel('Rate ({} spk / ms)'.format(r'$\tau$'), fontsize=fontsize)

    g = 4
    Erange = np.linspace(-1, Emax, 6)

    r_sim = []
    r_sim_stim = []

    J = np.random.binomial(n=1, p=connection_prob, size=(N,N)) * g / connection_prob / N

    for E in Erange:

        _, spktimes = sim_lif_pop(J=J, E=E, tstop=tstop, tstim=0, Estim=0)
        r_sim.append(len(spktimes) / N / tstop)
        
        _, spktimes = sim_lif_pop(J=J, E=E, tstop=tstop, tstim=tstim, Estim=Estim)
        
        spktimes = spktimes[spktimes[:, 0] > 2*tstim, :]
        if len(spktimes) > 0:
            r_sim_stim.append(len(spktimes) / N / (tstop - 2*tstim))
        else:
            r_sim_stim.append(np.nan)


    Erange_th = np.arange(-1, Emax, .01)

    r_th = [lif_rate_homog(g, E) for E in Erange_th]

    r_mft_low = 0 * Erange_th
    r_mft_low[Erange_th > 1] = np.nan

    v_mft_high = 0 * Erange_th
    r_mft_high = v_mft_high.copy()

    Ebif = np.where(Erange_th > g*(4-g)/4)[0][0]

    v_mft_high[Ebif:] = (g + np.sqrt(g**2 + 4*(Erange_th[Ebif:] - g))) / 2
    r_mft_high = intensity(v_mft_high)

    ax[1].plot(Erange, r_sim, 'ko', alpha=0.5)
    ax[1].plot(Erange, r_sim_stim, 'ko', alpha=0.5)
    ax[1].plot(Erange_th, r_mft_low, 'k', linewidth=2)
    ax[1].plot(Erange_th, r_mft_high, 'k--', linewidth=2)
    ax[1].plot(Erange_th, r_th, 'k', linewidth=2)

    ax[1].set_xlabel('E', fontsize=fontsize)
    ax[1].set_ylabel('Rate ({} spk / ms)'.format(r'$\tau$'), fontsize=fontsize)

    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(savefile)


if __name__ == '__main__':
    
    plot_fig_single_pop_weakly_coupled(savedir=results_dir)
    
    # plot_fig_single_pop_bifurcation()