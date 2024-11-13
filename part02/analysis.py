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
    dfs = []
    with zipfile.ZipFile(filename, 'r') as trg_zip:
        for item in trg_zip.namelist():
            if ds in item:
                with trg_zip.open(item) as trg_file:
                    dfs.extend(pd.read_html(trg_file, encoding='cp1250'))
    return pd.concat(dfs)

# Ukol 2: zpracovani dat


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    new_df = df.copy()
    
    new_df['date'] = pd.to_datetime(df['p2a'], dayfirst=True, yearfirst=False, format=None, errors='coerce')
    
    region_dict = {0: "PHA", 1: "STC", 2: "JHC", 3: "PLK", 4: "ULK", 5: "HKK",6:
    "JHM", 7: "MSK", 14: "OLK", 15: "ZLK", 16: "VYS", 17: "PAK", 18:
    "LBK", 19: "KVK"}
    new_df['region'] = new_df['p4a'].apply(lambda x: region_dict[x]) 

    new_df.drop_duplicates(subset=['p1'], inplace=True)
    
    if verbose:
        new_size = new_df.memory_usage(deep=True).sum()/1_000_000
        print(f'new_size={new_size} MB')
    
    return new_df

# Ukol 3: počty nehod v jednotlivých regionech podle stavu vozovky
def plot_state(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
    pass

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
