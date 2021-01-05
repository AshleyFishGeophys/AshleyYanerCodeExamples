import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import scipy.stats as stats
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

path = r'D:\MyStuff\bpx'

df = pd.read_csv(path + r'\GHG-emissions-data set_technical scenarioDec2020.csv')

df1 = df.melt('Well Site', var_name='months', value_name='% methane intensity')
df1['threshold'] = 10

sp = sns.scatterplot(x="Well Site", y="% methane intensity", hue='months', data=df1)
sp.set_ylabel('% methane intensity',fontsize=16)
sp.set_xlabel('Well Site',fontsize=16)
sp.set_title('Initial Look', fontsize=20)
sp.plot(df1["Well Site"],df1['threshold'], color='black')
plt.show()

# ----------------------------------- IQR -------------------------------

dfm = df.melt('Well Site', var_name='months', value_name='% methane intensity')
dfm['threshold'] = 0.2

b = sns.boxplot(data=dfm, x='Well Site', y ='% methane intensity', showmeans=True)
b.set_ylabel('% methane intensity',fontsize=16)
b.set_xlabel('Well Site',fontsize=16)
b.set_title('Analysis', fontsize=20)
plt.show()

df['mean'] = df.mean(axis=1)
df['std'] = df.std(axis=1, ddof=0)

# df_IQR = df[['Well Site','1/20', '2/20', '3/20', '4/20', '5/20', '6/20', '7/20']]
df_IQR = df[['Well Site']]
df_filt = df[['Well Site']]

# df_IQR['Q1'] = df.quantile(0.25, axis=1)
# df_IQR['Q3'] = df.quantile(0.75, axis=1)
df_IQR['IQR'] = abs(df.quantile(0.25, axis=1) - df.quantile(0.75, axis=1))
df_IQR['IQR_Q3'] = df.quantile(0.75, axis=1) + abs(df.quantile(0.25, axis=1) - df.quantile(0.75, axis=1))
df_IQR['IQR_Q1'] = df.quantile(0.25, axis=1) - abs(df.quantile(0.25, axis=1) - df.quantile(0.75, axis=1))

df_filt['1/20']=df['1/20'].where(df['1/20'].between(df_IQR['IQR_Q1'],df_IQR['IQR_Q3']))
df_filt['2/20']=df['2/20'].where(df['2/20'].between(df_IQR['IQR_Q1'],df_IQR['IQR_Q3']))
df_filt['3/20']=df['3/20'].where(df['3/20'].between(df_IQR['IQR_Q1'],df_IQR['IQR_Q3']))
df_filt['4/20']=df['4/20'].where(df['4/20'].between(df_IQR['IQR_Q1'],df_IQR['IQR_Q3']))
df_filt['5/20']=df['5/20'].where(df['5/20'].between(df_IQR['IQR_Q1'],df_IQR['IQR_Q3']))
df_filt['6/20']=df['6/20'].where(df['6/20'].between(df_IQR['IQR_Q1'],df_IQR['IQR_Q3']))
df_filt['7/20']=df['7/20'].where(df['7/20'].between(df_IQR['IQR_Q1'],df_IQR['IQR_Q3']))

# df_IQR.loc[df1['1/20'] > 1.5*df_IQR['IQR'], '1/20'] = 1
dfm_filt = df_filt.melt('Well Site', var_name='months', value_name='% methane intensity')
dfm_filt['mean_orig'] = df1.groupby('Well Site')['% methane intensity'].transform(np.nanmean)
dfm_filt['mean'] = dfm_filt.groupby('Well Site')['% methane intensity'].transform(np.nanmean)
dfm_filt['threshold'] = 0.2
print(dfm_filt.head(200))
# print(df_IQR.head(200))
# print(df_filt.head(200))



shut_ins = dfm_filt.query('mean > 0.2')
fig, ax = plt.subplots()
ax.set_title('Recommendations', fontsize=20)
ax.scatter(dfm_filt["Well Site"],dfm_filt['mean'],marker='*', s=100)
ax.plot(dfm_filt["Well Site"],dfm_filt['threshold'], color='black')
ax.scatter(shut_ins["Well Site"],shut_ins['mean'],marker='*', color="red", s=100)
# ax.scatter(dfm_filt["Well Site"],dfm_filt['mean_orig'])
ax.set_ylabel('mean(% methane intensity)',fontsize=16)
ax.set_xlabel('Well Site',fontsize=16)
ax.text(0.05, 3.2,'Remain on-line', color = 'blue', verticalalignment = 'top',fontsize=16)
ax.text(0.05, 3,'Recommended shut-in:', color = 'red', verticalalignment = 'top',fontsize=16)
ax.text(0.05, 0.35,'0.2% threshold', color = 'black', verticalalignment = 'top',fontsize=16)
ax.text(0.05, 2.85,'H, O-T, and FF-QQ', color = 'red', verticalalignment = 'top',fontsize=16)
plt.show()





# ------------------------------zscore --------------------------------------

# # calculate summary statistics
# df['mean'] = df.mean(axis=1)
# df['std'] = df.std(axis=1, ddof=0)
#
# df1 = df[['1/20', '2/20', '3/20', '4/20', '5/20', '6/20', '7/20']]
# df_zscore = df[['Well Site']]
#
# df_zscore['1/20'] = (df['1/20'] - df['mean']) / df['std']
# df_zscore['2/20'] = (df['2/20'] - df['mean']) / df['std']
# df_zscore['3/20'] = (df['3/20'] - df['mean']) / df['std']
# df_zscore['4/20'] = (df['4/20'] - df['mean']) / df['std']
# df_zscore['5/20'] = (df['5/20'] - df['mean']) / df['std']
# df_zscore['6/20'] = (df['6/20'] - df['mean']) / df['std']
# df_zscore['7/20'] = (df['7/20'] - df['mean']) / df['std']
#
# # for (columnName, columnData) in df1.iteritems():
# #     # col_zscore = columnName + '_zscore'
# #     col_zscore = columnName
# #     df_zscore[col_zscore] = (columnData - columnData.mean())/columnData.std(ddof=0)
#
# # print(df_zscore.head())
#
#
# # print(df_zscore.head())
# dfm_zscore = df_zscore.melt('Well Site', var_name='months', value_name='zscore')
#
# # print(dfm_zscore.head())
# df = df[['Well Site', '1/20', '2/20', '3/20', '4/20', '5/20', '6/20', '7/20']]
#
# dfm = df.melt('Well Site', var_name='months', value_name='% methane intensity')
#
# df_new = pd.concat([dfm, dfm_zscore], axis=1)
# # print(df_new.head())
#
# df_new = df_new[['Well Site', 'months', '% methane intensity', 'zscore']]
# df_new = df_new.loc[:, ~df_new.columns.duplicated()]
# df_new['threshold'] = 0.2

# df_new.loc[np.abs(df_new['zscore']) > 2, 'outlier'] = 'True'
# df_new.loc[np.abs(df_new['zscore']) < 2, 'outlier'] = 'False'

#
# sns.scatterplot(data=df_new,x="Well Site", y="% methane intensity", hue='months', style = 'outlier', markers=True)
# plt.show()


# df_new['% methane intensity']=df_new['% methane intensity'].where(df_new['zscore'].abs().between(0,2.5))
# df_new['mean'] = df_new.groupby('Well Site')['% methane intensity'].transform(np.mean)
#
# print(df_new.head())
# # sns.scatterplot(data=df_new,x="Well Site", y="% methane intensity", hue='months')
# plt.plot(df_new['Well Site'], df_new['threshold'])
# plt.scatter(df_new['Well Site'], df_new['mean'], marker='*', color='k')
# # Plot bars and create text labels for the table
# cell_text = []
# plt.show()








# ----------------------INITIAL LOOK -----------------------------------
# # df = df['Well Site', '1/20','2/20']
# # df = df[['Well Site','1/20','3/20','4/20','5/20']]
#
# df['avg'] = df.mean(axis=1)
# df['limit'] = 0.2
#
# # sns.lineplot(data = df, x = 'Well Site', y = 'avg')
# plt.fill_between(df['Well Site'], df['avg'], df['limit'], where=(df['avg'] >= df['limit']), color='r')
# plt.fill_between(df['Well Site'], df['avg'], df['limit'], where=(df['avg'] < df['limit']), color='b')
# plt.plot(df['Well Site'], df['limit'])
# plt.show()
#
# # print(df.head())
# #
# # sns.lineplot(data=df,  palette="tab10", linewidth=2.5)
# # plt.show()
#
# dfm = df.melt('Well Site', var_name='months',  value_name='% methane intensity')
# print(dfm.head())
#
# sns.catplot(x="Well Site", y="% methane intensity", hue='months', data=dfm, kind='point')
# plt.show()