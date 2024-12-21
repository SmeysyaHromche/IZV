import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd

matplotlib.use('TkAgg')


def age_num_to_age_group(age: int) -> str:
    """
    Convert string of age to age group
    Args:
        age:

    Returns:
        group of age
    """
    if age < 12:
        return 'Child'
    elif age >= 12 and age < 20:
        return 'Teenager'
    elif age >= 20 and age < 30:
        return 'Young'
    elif age >= 30 and age < 50:
        return 'Middle'
    elif age >= 50 and age < 65:
        return 'Late'
    else:
        return 'Old'


def get_substances_type(row) -> str:
    """
    Get substances by alcohol and drugs
    Args:
        row: row of dataframe

    Returns:
        substances type
    """
    if row['drug'] > 0 and row['drug'] < 7 and row['alcohol'] > 4:
        return 'Alcohol and Drug'
    elif row['drug'] > 0 and row['drug'] < 7:
        return 'Drug'
    else:
        return 'Alcohol'


def get_sex(value: int) -> str:
    """
    Get string format of sex
    Args:
        value:

    Returns:

    """
    if value == 1 or value == 3:
        return 'Male'
    else:
        return 'Female'


def load_data()->pd.DataFrame:
    """
    Load data from source and transform to usefully format
    Returns:
        data for analyse
    """
    # load data
    df_accidents = pd.read_pickle('accidents.pkl.gz')[['p1', 'p11a', 'region', 'date']]
    df_pedestrians = pd.read_pickle('pedestrians.pkl.gz')[['p1', 'p33d', 'p30a', 'p30b', 'p33c']]
    # merge to one dataframe
    df = pd.merge(df_accidents, df_pedestrians, on='p1', how='inner')

    # set substances
    df = df.rename(columns={'p30a': 'alcohol', 'p30b': 'drug'})
    df = df[((df['drug'] > 0) & (df['drug'] < 7)) | (df['alcohol'] > 4)]
    df['substances'] = df.apply(get_substances_type, axis=1)

    # set sex
    df['sex'] = df['p33c'].apply(get_sex)

    # set age
    df = df[~df['p33d'].isin(['XX', 'xx'])]
    df['p33d'] = pd.to_numeric(df['p33d'], errors='coerce')
    df = df.dropna(subset=['p33d'])
    df['age'] = df['p33d'].apply(age_num_to_age_group)

    # set month
    df['month'] = df['date'].dt.month_name()

    return df


def create_plot(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False) -> None:
    """
    Create groups of plots which represent drugs role in accidents with optional saving and displaying.
    Args:
        df: source data frame about accident.
        fig_location: path to save file. Defaults to None.
        show_figure:  optional flag about displaying. Defaults to False.

    Returns:
        None
    """
    # aux
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                   'November', 'December']

    # create grid place for plots
    fig = plt.figure(constrained_layout=True, figsize=(15, 12))
    spec = grd.GridSpec(nrows=2, ncols=3, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[0, 2])
    ax4 = fig.add_subplot(spec[1, :])

    # plot drugs relationships
    aux_drug = df.copy()
    aux_drug = aux_drug[aux_drug['substances'] == 'Drug']
    aux_drug['drug_type'] = aux_drug['drug'].map({1: 'THC', 2: 'AMP', 3: 'MET', 4: 'OPI', 5: 'BENZ', 6: 'ather'})

    # drugs and age
    aux_drug_age = aux_drug['age'].value_counts()
    ptch1, _, _ = ax1.pie(aux_drug_age, labels=None, autopct='%1.1f%%', startangle=90)
    ax1.legend(ptch1, aux_drug_age.index, title='Age groups', loc="best")
    ax1.set_title('Drugs and Age')
    # drugs and sex
    aux_drug_sex = aux_drug['sex'].value_counts()
    ptch2, _, _ = ax2.pie(aux_drug_sex, labels=None, autopct='%1.1f%%', startangle=90)
    ax2.legend(ptch2, aux_drug_sex.index, title='Sex groups', loc="best")
    ax2.set_title('Drugs and Sex')
    # type of drugs
    aux_drug_type = aux_drug['drug_type'].value_counts()
    ptch3, _, _ = ax3.pie(aux_drug_type, labels=None, autopct='%1.1f%%', startangle=90)
    ax3.legend(ptch3, aux_drug_type.index, title='Drugs groups', loc="best")
    ax3.set_title('Types of drugs')

    # generally data about substances and accidents
    df_aux_month = df.copy()
    df_aux_month = df_aux_month.groupby(['month', 'substances']).size().reset_index(name='count')
    df_aux_month['month'] = pd.Categorical(df_aux_month['month'], categories=month_order, ordered=True)
    month_stat = sns.barplot(data=df_aux_month, x='month', y='count', ax=ax4, hue='substances')
    month_stat.set_title('Alcohol vs Drugs by pedestrians throw months')
    month_stat.set_xlabel('Month')
    month_stat.set_ylabel('Accidents count')
    month_stat.legend(title='Substances')

    if show_figure:
        plt.show()
    if fig_location is not None:
        plt.savefig(fig_location)


def create_table(df: pd.DataFrame) -> None:
    """

    Args:
        df:

    Returns:

    """
    pass


def create_data_in_text(df: pd.DataFrame) -> None:
    """

    Args:
        df:

    Returns:

    """
    pass


if __name__ == '__main__':
    # p11a
    df = load_data()
    create_plot(df, 'fig.png')
    create_table(df)
    create_data_in_text(df)
