#!/usr/bin/env python3.12
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

# Ukol 1: nacteni dat ze ZIP souboru


def load_data(filename : str, ds : str) -> pd.DataFrame:
    '''
    Load, unit, clean target data from input file
    
    Args:
        filename (str): path to source file
        ds (str): name of trg xls file
    
    Returns:
        pd.DataFrame: Loaded data
    '''
    dfs = []
    with zipfile.ZipFile(filename, 'r') as trg_zip:
        for item in trg_zip.namelist():
            if ds in item:
                with trg_zip.open(item) as trg_file:  # check trg file
                    dfs.extend(pd.read_html(trg_file, encoding='cp1250'))
    df = pd.concat(dfs)  # all trg tabels to one
    return df.drop(columns=[col for col in df.columns if "Unnamed" in str(col)])  # drop unnamed columns

# Ukol 2: zpracovani dat


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    '''
    Filters and modifies input data into a form convenient for further analysis
    
    Args:
        df (pd.Dataframe):  input data in dataframe
        verbose (bool):     flag for size outpout data display

    Returns:
        pd.DataFrame: Transforms data
    '''
    REGION_DICT = {0: "PHA", 1: "STC", 2: "JHC", 3: "PLK", 4: "ULK", 5: "HKK",6:
    "JHM", 7: "MSK", 14: "OLK", 15: "ZLK", 16: "VYS", 17: "PAK", 18:
    "LBK", 19: "KVK"}
    
    new_df = df.copy()
    new_df['date'] = pd.to_datetime(df['p2a'], dayfirst=True, yearfirst=False, format=None, errors='coerce')
    new_df['region'] = new_df['p4a'].apply(lambda x: REGION_DICT[x])

    new_df.drop_duplicates(subset=['p1'], inplace=True)
    
    if verbose:
        new_size = new_df.memory_usage(deep=True).sum()/1_000_000
        print(f'new_size={new_size} MB')
    
    return new_df

# Ukol 3: počty nehod v jednotlivých regionech podle stavu vozovky
def plot_state(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False)->None:
    '''
    Create accident number chart and save it on target file with optional displaying.
    
    Args:
        df (pd.DataFrame): input data for accident.
        fig_location (str|None, optional): path/name for target file for saving. Defaults to None.
        show_figure (bool): flag for optional displaying. Defaults to False.

    Returns:
        None
    '''
    
    DICT_SURFACE_CONDITION = {1:'povrch suchý', 2:'povrch suchý', 3:'povrch mokrý', 4: 'na vozovce je bláto', 5:'na vozovce je náledí, ujetí sníh', 6:'na vozovce je náledí, ujetí sníh'}
    SURFACE_CONDITION = set(DICT_SURFACE_CONDITION.values())
    
    df_aux = df.copy()
    df_aux['p16'] = df_aux['p16'].replace(DICT_SURFACE_CONDITION)  # transform surface condition to trg string
    
    fig, axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True, figsize=(14,9))  # set chart
    fig.suptitle('Pocet nehod dle povrchu vozovky')
    for i, cond in enumerate(SURFACE_CONDITION):
        row = i // 2  # indexing by axes on chart
        col = i % 2
        
        trg_grouping = df_aux[df_aux['p16']==cond].groupby('region').count()['p1']  # statistics on accidents in the regions
        sns.barplot(ax=axes[row, col], x=trg_grouping.index, y=trg_grouping.values, hue=trg_grouping.index, zorder=2) # statistic chart
        
        # expected format
        axes[row, col].set_facecolor("lightblue")
        axes[row, col].grid(axis="y", color="red", linewidth=1, zorder=0)
        axes[row, col].tick_params(axis='x',width=0, length=0)
        axes[row, col].tick_params(axis='y',width=0, length=0)
        if col == 0:
            axes[row, col].set_ylabel('Pocet nehod')
        if row == 0:
            axes[row, col].tick_params(labelbottom=False)
        axes[row, col].title.set_text(f'Stav povrchu vozovky: {cond}')
    
    if show_figure:
        plt.show()
    if fig_location is not None:
        plt.savefig(fig_location)

# Ukol4: alkohol a následky v krajích
def plot_alcohol(df: pd.DataFrame, df_consequences : pd.DataFrame, 
                 fig_location: str = None, show_figure: bool = False):
    pass

# Ukol 5: Druh nehody (srážky) v čase
def plot_type(df: pd.DataFrame, fig_location: str = None,
              show_figure: bool = False):
    pass


if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.

    df = load_data("data_23_24.zip", "nehody")
    df_consequences = load_data("data_23_24.zip", "nasledky")
    df2 = parse_data(df, True)
    
    plot_state(df2, "01_state.png")
    plot_alcohol(df2, df_consequences, "02_alcohol.png", True)
    plot_type(df2, "03_type.png")

# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
