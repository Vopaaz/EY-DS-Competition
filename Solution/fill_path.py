import pandas as pd
import util

rDF = util.Raw_DF_Reader()
mainDF = rDF.test
hashSeries = mainDF['hash'].drop_duplicates()
hashSeries.index = list(range(0,len(hashSeries)))

'''
df = mainDF[mainDF['hash'] == '00032f51796fd5437b238e3a9823d13d_31']

for j in range(0,100):
    if(df.loc[j,'time_exit']!= df.loc[j+1,'time_entry']):

        insertRow = pd.DataFrame([[df.loc[j,'hash'],df.loc[j,'trajectory_id'],df.loc[j,'time_exit'],df.loc[j+1,'time_entry'],
                                   'nan','nan','nan',df.loc[j,'x_exit'],df.loc[j,'y_exit'],df.loc[j+1,'x_entry'],df.loc[j+1,'y_entry']]],
                                 columns=['hash', 'trajectory_id', 'time_entry', 'time_exit', 'vmax', 'vmin',
                                          'vmean', 'x_entry', 'y_entry', 'x_exit', 'y_exit'])

        #print(insertRow)
        above = df.loc[:j]
        below = df.loc[j+1:]
        df = above.append(insertRow,ignore_index = True).append(below,ignore_index = True)
    if(j >= len(df)-2):
        break


df.to_csv('temp.csv')
'''
'''
df = mainDF[mainDF['hash'] == '000479418b5561ab694a2870cc04fd43_25']
df.index = range(0,len(df))
for j in range(0, 100):
    if (j < len(df) - 1):
        if (df.loc[j, 'time_exit'] != df.loc[j + 1, 'time_entry']):
            insertRow = pd.DataFrame(
                [[df.loc[j, 'hash'], df.loc[j, 'trajectory_id'], df.loc[j, 'time_exit'], df.loc[j + 1, 'time_entry'],
                  'nan', 'nan', 'nan', df.loc[j, 'x_exit'], df.loc[j, 'y_exit'], df.loc[j + 1, 'x_entry'],
                  df.loc[j + 1, 'y_entry']]],
                columns=['hash', 'trajectory_id', 'time_entry', 'time_exit', 'vmax', 'vmin',
                         'vmean', 'x_entry', 'y_entry', 'x_exit', 'y_exit'])

            above = df.loc[:j]
            below = df.loc[j + 1:]
            df = above.append(insertRow, ignore_index=True).append(below, ignore_index=True)
    else:
        break

df.to_csv('temp.csv')
'''

fullDF = pd.DataFrame([], columns=['hash', 'trajectory_id', 'time_entry', 'time_exit', 'vmax', 'vmin',
                             'vmean', 'x_entry', 'y_entry', 'x_exit', 'y_exit'])
#grouped = mainDF.groupby(mainDF['hash'])
for i in mainDF['hash']:
    df = mainDF[mainDF['hash'] == i]
    df.index = range(0,len(df))
    for j in range(0, 100):
        if(j < len(df)-1):
            if (df.loc[j, 'time_exit'] != df.loc[j + 1, 'time_entry']):
                insertRow = pd.DataFrame(
                    [[df.loc[j, 'hash'], df.loc[j, 'trajectory_id'], df.loc[j, 'time_exit'], df.loc[j + 1, 'time_entry'],
                      'nan', 'nan', 'nan', df.loc[j, 'x_exit'], df.loc[j, 'y_exit'], df.loc[j + 1, 'x_entry'],
                      df.loc[j + 1, 'y_entry']]],
                    columns=['hash', 'trajectory_id', 'time_entry', 'time_exit', 'vmax', 'vmin',
                             'vmean', 'x_entry', 'y_entry', 'x_exit', 'y_exit'])

                above = df.loc[:j]
                below = df.loc[j + 1:]
                df = above.append(insertRow, ignore_index=True).append(below, ignore_index=True)
        else:
            break
    #print(df)
    fullDF = fullDF.append(df, ignore_index=True)
    #print(fullDF)

fullDF.to_csv('fullDF.csv')
