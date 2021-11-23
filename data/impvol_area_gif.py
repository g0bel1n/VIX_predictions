import re
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_impvol_area_points(spx_impvol: pd.DataFrame, regex: re.Pattern, i: int) -> tuple[list, list, list]:
    MTH = []
    STRIKE = []
    IMPVOL = []
    for col in spx_impvol.columns:
        if col != 'Date':
            resultat = regex.findall(col)
            MTH.append(int(resultat[0]))
            STRIKE.append(int(resultat[1]))
            IMPVOL.append(spx_impvol[col].iloc[i])
    return MTH, STRIKE, IMPVOL


def plot_impvol_area_at(i: int, regex: re.Pattern, spx_impvol: pd.DataFrame, name: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    MTH, STRIKE, IMPVOL = get_impvol_area_points(spx_impvol, regex, i)
    ax.plot_trisurf(MTH, STRIKE, IMPVOL, cmap=plt.cm.coolwarm, linewidth=0)
    ax.set_zlim3d([10.0, 35.0])
    ax.set_xlim3d([min(MTH), max(MTH)])
    ax.set_ylim3d([max(STRIKE), min(STRIKE)])
    ax.set_zlim3d([min(spx_impvol.min(skipna=True, numeric_only=True)), max(spx_impvol.max(skipna=True, numeric_only=True))])
    plt.xlabel('Maturity (in Months)')
    plt.ylabel('Strike Price (%of Stock Price)')
    ax.set_zlabel('Implied Volatility')
    plt.title(str(spx_impvol['Date'].iloc[i])[:-8])
    plt.savefig(f"3dPlots/{name}_{i}.png")
    plt.close()


def make_impvol_gif(spx_impvol: pd.DataFrame, name: str, duration=150, step=25) -> None:
    regex = re.compile("[0-9]+")
    # Generates the png's
    for i in tqdm(range(0, len(spx_impvol['Date']), step)):
        plot_impvol_area_at(i, regex, spx_impvol, name)

    # Join them into a gif
    images = [Image.open(f"3dPlots/{name}_{n}.png") for n in range(0, len(spx_impvol['Date']), step)]
    images[0].save(f'impvol_{name}.gif', save_all=True, append_images=images[1:], duration=duration, loop=0)
