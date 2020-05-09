"""
Loading utilities for computational experiments.
"""
import os
import pandas as pd


DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_zT(all_data=False):
    """
    Thermoelectric figures of merit for 165 experimentally measured compounds.

    Obtained from the Citrination database maintained by Citrine, Inc.
    Citrine obtained from Review https://doi.org/10.1021/cm400893e which took
    measurements at 300K from many original publications.

    All samples are
    - Measured at 300K (within 0.01 K)
    - polycrystalline


    If all_data is loaded, the columns are:
    - composition: composition as a string
    - zT: thermoelectric figure of merit
    - PF (W/m.K2): power factor
    - k (W/m.K): overall thermal conductivity
    - S (uV/K): Seebeck coefficient
    - log rho: Log resistivity, presumably in ohm-meters.

    Args:
        all_data (bool): Whether all data will be returned in the df. If False,
            only the compositions as strings and the zT measurements will be
            loaded.

    Returns:
        (pd.DataFrame): The dataframe containing the zT data.

    """
    path = os.path.join(DATA_DIR, "zT-citrination-165.csv")
    df = pd.read_csv(path, index_col=None)
    if not all_data:
        df = df[["composition", "zT"]]
    return df


def load_e_form():
    """
    85,014 DFT-GGA computed formation energies.

    Ground state formation energies from the Materials Project, adapted
    from https://github.com/CJBartel/TestStabilityML/blob/master/mlstabilitytest/mp_data/data.py
    originally gathered from the Materials Project via MAPI on Nov 6, 2019.

    There is exactly one formation energy per composition. The formation energy
    was chosen as the ground state energy among all sructures with the desired
    composition.

    Returns:
        (pd.DataFrame): The formation energies and compositions
    """
    path = os.path.join(DATA_DIR, "eform-materialsproject-85014.csv")
    df = pd.read_csv(path, index_col="mpid")
    return df


def load_bandgaps():
    """
    4,604 experimental band gaps, one per composition.

    Matbench v0.1 test dataset for predicting experimental band gap from
    composition alone. Retrieved from Zhuo et al
    (https:doi.org/10.1021/acs.jpclett.8b00124) supplementary information.
    Deduplicated according to composition, removing compositions with reported
    band gaps spanning more than a 0.1eV range; remaining compositions were
    assigned values based on the closest experimental value to the mean
    experimental value for that composition among all reports.

    Returns:
        (pd.DataFrame): Experimental band gaps and compositions as strings

    """
    path = os.path.join(DATA_DIR, "bandgap-zhuo-4604.csv")
    df = pd.read_csv(path, index_col=False)
    return df


def load_steels():
    """
    312 yeild strengths of various steels.

    Matbench v0.1 dataset for predicting steel yield strengths from chemical
    composition alone. Retrieved from Citrine informatics. Deduplicated.

    Experimentally measured steel yield strengths, in GPa.
    https://citrination.com/datasets/153092/

    Returns:
        (pd.DataFrame): Dataframe of yield strengths per composition.
    """
    path = os.path.join(DATA_DIR, "yieldstrength-citrination-312.csv")
    df = pd.read_csv(path, index_col=False)
    return df


if __name__ == "__main__":
    df = load_steels()
    print(df)