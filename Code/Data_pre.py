
import pandas as pd
import numpy as np

import pandas as pd
import openpyxl
from openpyxl import load_workbook

df = pd.read_csv('DATA_Origin.csv')
print(df)


nrows = df.shape[0]  # 最大行数
print(nrows)
#总共需要多少excel

limit=20
sheets = nrows // limit
print(sheets)

for i in range(sheets):
    star=i*20
    df1=df[star:star+20]
    #print(chunk)
    df1.to_csv(f'{i+1}.csv',index=False)
