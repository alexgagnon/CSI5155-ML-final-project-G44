import pandas as pd
import plotly_express as px


def print_df_meta(df, label='', summarize=False):
    """Prints the pandas dataframe information"""
    print()
    print(label)
    if (summarize):
        print(df.info())
    else:
        print('--- INFO ---')
        print(df.info())
        print()
        print('--- DESCRIBE ---')
        print(df.describe())
        print()
        print('--- HEAD ---')
        print(df.head())
        print()


def report_cluster(X_train, y_test, y_pred):
    correct = 0

    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = kmeans.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1

    print(correct/len(X))


def null_analysis(df):
    '''
    desc: get nulls for each column in counts & percentages
    arg: df
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
