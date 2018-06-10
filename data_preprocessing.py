# df = Pandas DataFrame

#Create a new column from conditions on other columns
df['column_y'] = df[(df['column_x1'] | 'column_x2') & 'column_x3']
df['column_y'] = df['column_y'].apply(bool)
df['column_y'] = df['column_y'].apply(int)

#Create a new True/False column according to the first letter on another column.
lEI = [0] * df.shape[0]

for i, row in df.iterrows():
    try:
        l = df['room_list'].iloc[i].split(', ')
    except:
        #When the given row is empty
        l = []
    for element in l:
        if element[0] =='E' or element[0] == 'I':
            lEI[i] = 1

df['EI'] = pd.Series(lEI)
