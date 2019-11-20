import pandas as pd
import plotly_express as px


def print_df_meta(dataframe):
    """Prints the pandas dataframe information"""
    print('--- INFO ---')
    print(dataframe.info())
    print()
    print('--- DESCRIBE ---')
    print(dataframe.describe())
    print()
    print('--- HEAD ---')
    print(dataframe.head())
    print()

    # visualise null table
    null_table = null_analysis(dataframe)
    px.bar(null_table.reset_index(), x='index',
           y='percentage', text='counts', height=500)


def null_analysis(df):
    '''
    desc: get nulls for each column in counts & percentages
    arg: dataframe
    return: dataframe
    '''
    null_cnt = df.isnull().sum()  # calculate null counts
    null_cnt = null_cnt[null_cnt != 0]  # remove non-null cols
    null_percent = null_cnt / len(df) * 100  # calculate null percentages
    null_table = pd.concat(
        [pd.DataFrame(null_cnt), pd.DataFrame(null_percent)], axis=1)
    null_table.columns = ['counts', 'percentage']
    null_table.sort_values('counts', ascending=False, inplace=True)
    return null_table
