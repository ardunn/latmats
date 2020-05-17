import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
        {"name": "e_form", "title": r'DFT-PBE $E_f$', "y_label": r'$E_f$ (eV/atom)', "x_label": r'$E_f$ predicted (eV/atom)', "true": e_form_true, "pred": e_form_pred},
        {"name":"expt_gaps", "title": r'Experimental $E_g$', "y_label": r'$E_g$ (eV)', "x_label": r'$E_g$ predicted (eV)', "true": bg_true, "pred": bg_pred},
        {"name": "zT", "title": "Experimental Thermoelectric zT", "y_label": r'zT', "x_label": r'zT predicted', "true": zt_true, "pred": zt_pred}
    ]

    f, axes = plt.subplots(1, 3)

    for i, ax in enumerate(axes):
        d = data[i]
        true = d["true"].tolist()
        pred = d["pred"].tolist()

        maxtrue = max(true)
        mintrue = min([min(true), 0])

        maxtot = max(true + pred)
        mintot = min([min(true + pred), 0])

        sns.scatterplot(y=true, x=pred, ax=ax)
        ax.plot([mintrue, maxtrue], [mintrue, maxtrue], color="black")
        ax.set_ylim((mintot, maxtot))
        ax.set_xlim((mintot, maxtot))
        ax.set_xlabel(d["x_label"])
        ax.set_ylabel(d["y_label"])
        ax.set_aspect("equal", "box")



    return plt



if __name__ == "__main__":
    plt = graph_one2one()
    plt.show()
