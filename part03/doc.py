import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from typing import Tuple
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


def get_substances_type(row:pd.Series) -> str:
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


def load_data()->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from source and transform to usefully format
    Returns:
        data for analyse
    """
    # load data
    df_accidents = pd.read_pickle('accidents.pkl.gz')[['p1', 'p11a', 'region', 'date', 'p14*100']]
    df_pedestrians = pd.read_pickle('pedestrians.pkl.gz')[['p1', 'p33d', 'p30a', 'p30b', 'p33c', 'p33g']]
    # merge to one dataframe
    df = pd.merge(df_accidents, df_pedestrians, on='p1', how='inner')
    df_substances = df.copy()

    # set substances
    df_substances = df_substances.rename(columns={'p30a': 'alcohol', 'p30b': 'drug', 'p33d':'age'})
    df_substances = df_substances[((df_substances['drug'] > 0) & (df_substances['drug'] < 7)) | (df_substances['alcohol'] > 4)]
    df_substances['substances'] = df_substances.apply(get_substances_type, axis=1)

    # set sex
    df_substances['sex'] = df_substances['p33c'].apply(get_sex)

    # set age
    df_substances = df_substances[~df_substances['age'].isin(['XX', 'xx'])]
    df_substances['age'] = pd.to_numeric(df_substances['age'], errors='coerce')
    df_substances = df_substances.dropna(subset=['age'])
    df_substances['age_group'] = df_substances['age'].apply(age_num_to_age_group)

    # set month
    df_substances['month'] = df_substances['date'].dt.month_name()

    return df, df_substances


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
    aux_drug['drug type'] = aux_drug['drug'].map({1: 'THC', 2: 'AMP', 3: 'MET', 4: 'OPI', 5: 'BENZ', 6: 'ather'})

    # drugs and age
    aux_drug_age = aux_drug['age_group'].value_counts()
    ptch1, _, _ = ax1.pie(aux_drug_age, labels=None, autopct='%1.1f%%', startangle=90)
    ax1.legend(ptch1, aux_drug_age.index, title='Age groups', loc="best")
    ax1.set_title('Drugs and Age')
    # drugs and sex
    aux_drug_sex = aux_drug['sex'].value_counts()
    ptch2, _, _ = ax2.pie(aux_drug_sex, labels=None, autopct='%1.1f%%', startangle=90)
    ax2.legend(ptch2, aux_drug_sex.index, title='Sex groups', loc="best")
    ax2.set_title('Drugs and Sex')
    # type of drugs
    aux_drug_type = aux_drug['drug type'].value_counts()
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


def create_table(df: pd.DataFrame, ltx_table:str=None, show_table:bool=False) -> None:
    """
    Creates a table describing the average pedestrian victim under the influence of a particular type of drug.

    Args:
        df: input data
        ltx_table: flag for creating latex table
        show_table: flag for display table in text format

    Returns:
        None
    """
    df_aux = df[df['substances'] == 'Drug'].copy()
    df_aux['drug type'] = df_aux['drug'].map({1: 'THC', 2: 'AMP', 3: 'MET', 4: 'OPI', 5: 'BENZ', 6: 'ather'})
    df_aux = df_aux[['drug type', 'region', 'sex', 'month', 'age']]

    age_mean_by_drug = df_aux.groupby('drug type')['age'].mean().round().astype(int)
    table = df_aux.groupby('drug type').agg(lambda x: x.mode()[0]).reset_index()
    table['age'] = table['drug type'].map(age_mean_by_drug)

    if ltx_table:  # save latex table
        latx = table.to_latex(caption='Average pedestrian victim under the influence of a particular type of drug',
                          bold_rows=True)
        with open(f'{ltx_table}.tex', 'w') as f:
            f.write(latx)

    if show_table:  # show table
        print('==========================================================================')
        print('Average pedestrian victim under the influence of a particular type of drug')
        print('==========================================================================')
        print(table)
        print('==========================================================================')
        print()
        print()


def create_data_in_text(df: pd.DataFrame, df_substances:pd.DataFrame) -> None:
    """
    Create statistics data which will be used in report.
    Args:
        df: full data
        df_substances: data specified for substances representation

    Returns:

    """
    drugs_ped = df_substances[df_substances['substances'] == 'Drug'].size/df.size*100
    drugs_ped = round(drugs_ped)
    drugs_ped_death = df_substances[(df_substances['substances'] == 'Drug') & df_substances['p33g']==1].size/df[df['p33g']==1].size*100
    drugs_ped_death = round(drugs_ped_death)
    drugs_ped_avg_loss = df_substances[df_substances['substances'] == 'Drug']['p14*100'].agg(['min', 'max', 'mean']).round().astype(int)

    print(f'The ratios of drug-involved pedestrian accidents to all accidents: {drugs_ped} %')
    print(f'The ratio of drug-involved pedestrian fatality accidents to all pedestrian fatality accidents: {drugs_ped_death} %')
    print(f'The minimum loss by accident with drug-involved pedestrian (CZ): {drugs_ped_avg_loss['min']}')
    print(f'The mean loss by accident with drug-involved pedestrian (CZ): {drugs_ped_avg_loss['mean']}')
    print(f'The maximum loss by accident with drug-involved pedestrian (CZ): {drugs_ped_avg_loss['max']}')
    print()
    print()


if __name__ == '__main__':
    df, df_substances = load_data()
    create_plot(df_substances, 'fig.png')
    create_table(df_substances, show_table=True, ltx_table='table')
    create_data_in_text(df, df_substances)
