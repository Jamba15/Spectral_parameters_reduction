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


def load_results_sparse(perc, ris_folder, lay_type):
    ris = []
    for p in perc:
        tmp = []
        with open(join(ris_folder, lay_type + '_' + str(p) + '.p'), 'rb') as f:
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

    if ris_folder.find('CIFAR') == -1:
        N1 = 28*28
    else:
        N1 = 32*32*3

    NumParD = np.arange(1, Rb) * N1 + np.arange(1, Rb) * 10 + np.arange(1, Rb) + 10
    NumParSSVD = [np.min((i, N1)) + np.min((i, 10)) + N1 + 3 * i + 10 + 10 for i in range(1, Rb)]
    NumParSpCompl = N1 + 2 * np.arange(1, Rb) + 10 + np.arange(1, Rb) + 10
    NumParQR = [i + N1 + i + 10 + np.min((i, N1)) * np.min((i, N1)) / 2 + np.min((i, N1)) / 2 + np.min((i, 10)) * np.min(
                (i, 10)) / 2 + np.min((i, 10)) / 2 for i in range(1, Rb)]



    rhoSpCompl = np.array(NumParSpCompl) / np.array(NumParD)
    rhoSSVD = np.array(NumParSSVD) / np.array(NumParD)
    rhoQR = np.array(NumParQR) / np.array(NumParD)

    col = ['tab:blue', 'tab:olive', 'tab:red']
    leg = ['Spectral', 'S-SVD', 'QR']

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

    ax[1].set_xlabel(r'$N_2$', fontsize=25)
    ax[0].set_ylabel(r'Relative accuracy', rotation=90, fontsize=27, labelpad=15)

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
    ax[1].plot(n, rhoSpCompl[[list(n - 1)]], color=col[3], marker='o', markersize=ms1, linestyle='none')
    ax[1].set_ylabel(r'$\rho$', fontsize=f, rotation=0, labelpad=20)

    ax[0].set_xlim(Lb - 1, Rb - 80)
    ax[1].set_xlim(Lb - 1, Rb - 80)

    ax[0].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    plt.tight_layout()

    plt.show()


def res_plot_sparse(percentiles, ris_folder):
    p = np.array(percentiles)

    Dense = load_results_sparse(perc=percentiles,
                                ris_folder=ris_folder,
                                lay_type='Dense')
    QR = load_results_sparse(perc=percentiles,
                             ris_folder=ris_folder,
                             lay_type='QR_sparse')

    Dense_acc = []
    Dense_err = []
    QR_acc = []
    QR_err = []

    for i in range(len(p)):
        Dense_acc.append(Dense[i][0])
        Dense_err.append(Dense[i][1])
        QR_acc.append(QR[i][0])
        QR_err.append(QR[i][1])
    Dense_acc = np.array(Dense_acc)
    Dense_err = np.array(Dense_err)
    QR_acc = np.array(QR_acc)
    QR_err = np.array(QR_err)

    plt.rcParams.update({'font.size': 23})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    ax.plot(p, Dense_acc, color='k', marker='d')
    ax.fill_between(p, list(Dense_acc + Dense_err), list(Dense_acc - Dense_err), color='k', alpha=0.3)
    ax.plot(p, QR_acc, color='tab:red', marker='p')
    ax.fill_between(p, list(QR_acc + QR_err), list(QR_acc - QR_err), color='tab:red', alpha=0.3)

    ax.set_xlim([0.75, 1.0])

    ax.set_ylabel(r'Accuracy', fontsize=25, rotation=90, labelpad=15)
    ax.set_xlabel(r'Sparsity', fontsize=25, rotation=0, labelpad=15)
    plt.tight_layout()

    plt.show()
