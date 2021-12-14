# %%
import os
import pandas as pd
from pathlib import Path


if not "root" in locals():
    current_path = Path(os.getcwd())
    root = current_path.parent.absolute()
    data_folder = str(root) + '\\treated_data\\'


VIX = "PX_OPEN_VIX_volatility"


def create_box(df, threshold = 2., box_length = 7):
    for i in range(box_length):
        df["VIX_sup_" + str(i)] = (df[VIX].shift(i) > df[VIX] + threshold) # Est-ce que le i-ème jour est au dessus du threshold haut ?
        df["VIX_inf_" + str(i)] = (df[VIX].shift(i) < df[VIX] - threshold) # Est-ce que le i-ème jour est en dessous du threshold bas ?
        

    # Initialisation : Est-ce que le dès le lendemain, on dépasse (1), on est en dessous (-1) ou on est dans la boite (0) ?
    df["Box"] = 1*df["VIX_sup_1"] + (-1) * df["VIX_inf_1"] 


    for i in range(2, box_length):
        # Iterations : Si la valeur n'est pas encore sortie (Box == 0), sort-elle par le haut (1), bas (-1) ou pas du tout (0) de la boite au jour i ?
        df["Box"] = (df["Box"] == 0) * (1 *df[f"VIX_sup_{i}"] + (-1) * df[f"VIX_inf_{i}"]) + (df["Box"] != 0) * df["Box"]

    df = df.drop(columns = ["VIX_sup_" + str(i) for i in range(box_length)] + ["VIX_inf_" + str(i) for i in range(box_length)])
    return df