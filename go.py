import numpy
import pandas


df_list = []
for i in range(6):
    p = rf'C:\Users\QZQ\Desktop\实验结果存档\gmm single dnn\gmm_head_{i}\gmm_head_{i}_metrics.csv'
    df = pandas.read_csv(p, index_col=0)
    df_list.append(df.T)
df = pandas.concat(df_list)
print(df)

record = {}
for col in ['rec', 'acc', 'prc_auc', 'mcc', 'g_means']:
    record.update({col: [numpy.average(df[col])]})
print(record)
df = pandas.DataFrame(record)
df.to_csv('ave_metrics.csv')