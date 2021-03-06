# %%
import os
import pandas as pd
from pathlib import Path
if not "root" in locals():
    current_path = Path(os.getcwd())
    root = current_path.parent.absolute()
    data_folder = str(root) + '\\treated_data\\'

VIX = "PX_OPEN_VIX_volatility"


def create_box(df: pd.DataFrame, threshold=2., box_length=7, relative_threshold=None) -> pd.DataFrame:

    if relative_threshold:
        for i in range(-1, -box_length, -1):
            # Est-ce que le i-ème jour est au dessus du threshold haut ?
            df["VIX_sup_" + str(i)] = (df[VIX].shift(i) > df[VIX] * (1. + relative_threshold))
            df["VIX_inf_" + str(i)] = (df[VIX].shift(i) < df[VIX] * (1. - relative_threshold))

    else:
        for i in range(-1, -box_length, -1):
            df["VIX_sup_" + str(i)] = (df[VIX].shift(i) > df[
                VIX] + threshold)  # Est-ce que le i-ème jour est au dessus du threshold haut ?
            df["VIX_inf_" + str(i)] = (df[VIX].shift(i) < df[
                VIX] - threshold)  # Est-ce que le i-ème jour est en dessous du threshold bas ?

    # Initialisation : Est-ce que le dès le lendemain, on dépasse (1), on est en dessous (-1) ou on est dans la boite
    # (0) ?
    df["Box"] = 1 * df["VIX_sup_-1"] + (-1) * df["VIX_inf_-1"]

    for i in range(-2, -box_length, -1):
        # Iterations : Si la valeur n'est pas encore sortie (Box == 0), sort-elle par le haut (1), bas (-1) ou pas du
        # tout (0) de la boite au jour i ?
        df["Box"] = (df["Box"] == 0) * (1 * df[f"VIX_sup_{i}"] + (-1) * df[f"VIX_inf_{i}"]) + (df["Box"] != 0) * df[
            "Box"]

    df = df.drop(
        columns=["VIX_sup_" + str(i) for i in range(box_length)] + ["VIX_inf_" + str(i) for i in range(-1, -box_length, -1)])
    return df


def create_binary_box(df: pd.DataFrame, threshold=2., box_length=7, relative_threshold = None) -> pd.DataFrame:

    if relative_threshold:
        for i in range(-1, -box_length, -1):
            df["VIX_sup_" + str(i)] = (df[VIX].shift(i) > df[VIX] *(1+relative_threshold))

    else:
        for i in range(-1, -box_length, -1):
            df["VIX_sup_" + str(i)] = (df[VIX].shift(i) > df[
                VIX] + threshold)  # Est-ce que le i-ème jour est au dessus du threshold haut ?

    # Initialisation : Est-ce que le dès le lendemain, on dépasse (1), on est en dessous (-1) ou on est dans la boite
    # (0) ?
    df["Box"] = 1 * df["VIX_sup_-1"]

    for i in range(-1, -box_length, -1):
        # Iterations : Si la valeur n'est pas encore sortie (Box == 0), sort-elle par le haut (1), bas (-1) ou pas du
        # tout (0) de la boite au jour i ?
        df["Box"] = (df["Box"] == 0) * (1 * df[f"VIX_sup_{i}"]) + (df["Box"] != 0) * df["Box"]

    df = df.drop(columns=["VIX_sup_" + str(i) for i in range(-1, -box_length, -1)])
    return df

def create_binary_box_other_col(df: pd.DataFrame, threshold=2., box_length=7, col =VIX, relative_threshold = None) -> pd.DataFrame:

    if relative_threshold:
        for i in range(-1, -box_length, -1):
            df["VIX_sup_" + str(i)] = (df[col].shift(i) > df[col] *(1+relative_threshold))

    else:
        for i in range(-1, -box_length, -1):
            df["VIX_sup_" + str(i)] = (df[col].shift(i) > df[
                col] + threshold)  # Est-ce que le i-ème jour est au dessus du threshold haut ?

    # Initialisation : Est-ce que le dès le lendemain, on dépasse (1), on est en dessous (-1) ou on est dans la boite
    # (0) ?
    df["Box"] = 1 * df["VIX_sup_-1"]

    for i in range(-1, -box_length, -1):
        # Iterations : Si la valeur n'est pas encore sortie (Box == 0), sort-elle par le haut (1), bas (-1) ou pas du
        # tout (0) de la boite au jour i ?
        df["Box"] = (df["Box"] == 0) * (1 * df[f"VIX_sup_{i}"]) + (df["Box"] != 0) * df["Box"]

    df = df.drop(columns=["VIX_sup_" + str(i) for i in range(-1, -box_length, -1)])
    return df


def create_binary_box_end(df: pd.DataFrame, threshold=2., box_shift=7) -> pd.DataFrame:

    df["Box"] = (df[VIX].shift(-box_shift) > df[VIX] * (1+threshold))
    
    return df