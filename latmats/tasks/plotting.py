import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text', usetex=True)


def graph_one2one():
    dfef = pd.read_csv("comptexnet_e_form_test.csv")
    e_form_true = dfef["e_form (eV/atom)"]
    e_form_pred = dfef["predicted e_form (eV/atom)"]

    dfbg = pd.read_csv("comptexnet_expt_gaps_test.csv")
    bg_true = dfbg["bandgap (eV)"]
    bg_pred = dfbg["predicted bandgap (eV)"]

    dfzt = pd.read_csv("comptexnet_zT_test.csv")
    zt_true = dfzt["zT"]
    zt_pred = dfzt["predicted zT"]

    data = [
        {"name": "e_form", "title": r'DFT-PBE $E_f$', "y_label": r'$E_f$ (eV/atom)', "x_label": r'$E_f$ predicted (eV/atom)', "true": e_form_true, "pred": e_form_pred, "markersize": 3},
        {"name":"expt_gaps", "title": r'Experimental $E_g$', "y_label": r'$E_g$ (eV)', "x_label": r'$E_g$ predicted (eV)', "true": bg_true, "pred": bg_pred, "markersize": 8},
        {"name": "zT", "title": "Experimental Thermoelectric zT", "y_label": r'zT', "x_label": r'zT predicted', "true": zt_true, "pred": zt_pred, "markersize": 20}
    ]

    f, axes = plt.subplots(1, 3)

    for i, ax in enumerate(axes):
        d = data[i]
        true = d["true"].tolist()
        pred = d["pred"].tolist()

        maxtrue = max(true)
        mintrue = min([min(true), 0])

        maxtot = max(true + pred) * 1.1
        mintot = min([min(true + pred), 0]) * 0.9


        # Color by deviation
        hues = np.power(np.abs(np.asarray(true) - np.asarray(pred)), 0.5)
        hues_mean = np.mean(hues)
        hues_std = np.std(hues)
        hues = (hues - hues_mean)/hues_std

        # pal5 = sns.blend_palette(["blue", "green", "red"], as_cmap=True)
        pal5 = "magma_r"

        sns.scatterplot(y=true, x=pred, ax=ax, s=d["markersize"], hue=hues, palette=pal5, legend=False, linewidth=0, alpha=1.0)
        ax.plot([mintrue, maxtrue], [mintrue, maxtrue], color="black")
        ax.set_ylim((mintot, maxtot))
        ax.set_xlim((mintot, maxtot))

        fontsize = "x-large"
        ax.set_xlabel(d["x_label"], fontsize=fontsize)
        ax.set_ylabel(d["y_label"], fontsize=fontsize)
        ax.set_aspect("equal", "box")
        ax.set_title(d["title"], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=14)

    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(hspace=0.3)  # set the spacing between axes.

    # plt.tight_layout(wpad=-2)

    return plt


def graph_learning_rate():

    n_samples = [100, 500, 1000, 5000, 10000, 51008]
    maes_dense =  [ 0.75828, 0.49260501, 0.44516, 0.287502, 0.2099714, 0.12084700]
    maes_attn = [1.545, 0.662, 0.4463, 0.2353, 0.1562, 0.0859]

    series = {
        "densenet": maes_dense,
        "attention": maes_attn
    }
    fig, ax = plt.subplots()

    for name, data in series.items():
        label = "1-layer Attention" if name == "attention" else "2-layer DenseNet"
        ax.plot(n_samples, data, marker='o', linestyle='--', linewidth=2, markersize=6, label=label)

    fontsize = "x-large"
    ax.legend(fontsize=fontsize)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r'Log training set size', fontsize=fontsize)
    ax.set_ylabel(r'Log MAE', fontsize=fontsize)
    ax.set_title(r'Effect of training set size on error rate for $E_f$ prediction', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=18)
    return plt


if __name__ == "__main__":
    # plt = graph_one2one()
    plt = graph_learning_rate()
    plt.show()
