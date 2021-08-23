import pickle
import matplotlib.pyplot as plt
import numpy as np
from os.path import join


def load_results(nlist, ris_folder, lay_type):
    ris = []
    for n in nlist:
        tmp = []
        with open(join(ris_folder, lay_type + str(n) + '.p'), 'rb') as f:
            while True:
                try:
                    tmp.append(pickle.load(f))
                except EOFError:
                    break
            tmp = np.array(tmp)
        ris.append([tmp.mean(axis=0), tmp.var(axis=0)])
    return ris

def plot_results(n, ris_folder):

    Dense = load_results(nlist=n, ris_folder=ris_folder, lay_type='Dense')
    SpecFull = load_results(nlist=n, ris_folder=ris_folder, lay_type='Spectral_full')
    QR = load_results(nlist=n, ris_folder=ris_folder, lay_type='QR')
    SSVD = load_results(nlist=n, ris_folder=ris_folder, lay_type='SSVD_full')

    acc_dense = np.array([Dense[i][0] for i in range(len(n))])
    acc_spectral_full = np.array([SpecFull[i][0] for i in range(len(n))])
    acc_qr = np.array([QR[i][0] for i in range(len(n))])
    acc_ssvd = np.array([SSVD[i][0] for i in range(len(n))])

    err_dense = np.array([Dense[i][1] for i in range(len(n))])
    err_spectral_full = np.array([SpecFull[i][1] for i in range(len(n))])
    err_qr = np.array([QR[i][1] for i in range(len(n))])
    err_ssvd = np.array([SSVD[i][1] for i in range(len(n))])

    rel_acc_specfull = acc_spectral_full / acc_dense
    rel_acc_ssvd = acc_ssvd / acc_dense
    rel_acc_qr = acc_qr / acc_dense

    err_rel_acc_specfull = (err_spectral_full / acc_spectral_full + err_dense / acc_dense) * rel_acc_specfull
    err_rel_acc_ssvd = (err_ssvd / err_ssvd + err_dense / acc_dense) * rel_acc_ssvd
    err_rel_acc_qr = (err_qr / acc_qr + err_dense / acc_dense) * rel_acc_qr

    # Parameters count

    ms = 9
    ms1 = 7

    Rb = n[-1] + 100
    Lb = n[0] - 5

    N1 = 32 * 32 * 3

    NumParD = np.arange(1, Rb) * N1 + np.arange(1, Rb) * 10 + np.arange(1, Rb) + 10
    NumParS = []
    for i in range(1, Rb):
        NumParS.append(np.min((i, N1)) + np.min((i, 10)) + N1 + 3 * i + 10 + 10)
    NumParSp = np.arange(1, Rb) + 10 + np.arange(1, Rb) + 10
    NumParSpCompl = N1 + 2 * np.arange(1, Rb) + 10 + np.arange(1, Rb) + 10
    NumParQR = []
    for i in range(1, Rb):
        NumParQR.append(
            i + N1 + i + 10 + np.min((i, N1)) * np.min((i, N1)) / 2 + np.min((i, N1)) / 2 + np.min((i, 10)) * np.min(
                (i, 10)) / 2 + np.min((i, 10)) / 2)

    rhoSp = np.array(NumParSp) / np.array(NumParD)
    rhoSpCompl = np.array(NumParSpCompl) / np.array(NumParD)
    rhoSSVD = np.array(NumParS) / np.array(NumParD)
    rhoQR = np.array(NumParQR) / np.array(NumParD)

    col = ['tab:blue', 'tab:olive', 'tab:red']
    leg = ['Spectral', 'S-SVD', 'QR']
    f = 25
    f1 = 15
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plt.rcParams.upacc_densete({'font.size': 23})

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ax[0].plot(np.arange(0, Rb), np.ones(int(Rb)), color='k', linestyle='acc_denseshed', linewidth=2)

    ax[0].plot(n, rel_acc_specfull, marker='^', markersize=ms, color=col[0], label=leg[0])
    ax[0].plot(n, rel_acc_ssvd, marker='p', markersize=ms, color=col[1], label=leg[1])
    ax[0].plot(n, rel_acc_qr, marker='d', markersize=ms, color=col[2], label=leg[2])

    ax[0].fill_between(n,
                       (rel_acc_specfull - err_rel_acc_specfull).astype(float),
                       (rel_acc_specfull + err_rel_acc_specfull).astype(float),
                       color=col[0],
                       alpha=0.3)
    ax[0].fill_between(n,
                       (rel_acc_ssvd - err_rel_acc_ssvd).astype(float),
                       (rel_acc_ssvd + err_rel_acc_ssvd).astype(float),
                       color=col[1],
                       alpha=0.3)
    ax[0].fill_between(n,
                       (rel_acc_qr - err_rel_acc_qr).astype(float),
                       (rel_acc_qr + err_rel_acc_qr).astype(float),
                       color=col[2],
                       alpha=0.3)

    ax[1].set_xlabel(r'$N_2$', fontsize=f)
    ax[0].set_ylabel(r'Relative accuracy', rotation=90, fontsize=f + 2, labelpad=15)

    ax[0].tick_params(
        axis='x',  # changes apply to the x-axis
        length=4,
        width=1,
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    ax[1].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',
        length=4,
        width=1)

    ax[1].plot(np.arange(1, Rb), rhoSSVD, color=col[4], linestyle='acc_denseshed')
    ax[1].plot(np.arange(1, Rb), rhoQR, color=col[5], linestyle='acc_denseshed')
    ax[1].plot(np.arange(1, Rb), rhoSpCompl, color=col[3], linestyle='acc_denseshed')

    n = np.array(n)
    ax[1].plot(n, rhoSSVD[[list(n - 1)]], color=col[4], marker='o', markersize=ms1, linestyle='none')
    ax[1].plot(n, rhoQR[[list(n - 1)]], color=col[5], marker='o', markersize=ms1, linestyle='none')
    # ax[1].plot(n,rhoSp[[list(n-1)]],color=col[1], marker='o', markersize=ms1, linestyle='none')
    ax[1].plot(n, rhoSpCompl[[list(n - 1)]], color=col[3], marker='o', markersize=ms1, linestyle='none')
    ax[1].set_ylabel(r'$\rho$', fontsize=f, rotation=0, labelpad=20)

    ax[0].set_xlim(Lb - 1, Rb - 80)
    ax[1].set_xlim(Lb - 1, Rb - 80)

    ax[0].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    plt.tight_layout()

    # plt.savefig('FigureArticolo/RelAcc_ML_Fm')
    plt.show()
