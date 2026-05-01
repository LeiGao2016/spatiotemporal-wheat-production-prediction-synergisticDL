import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error
from sklearn.metrics import r2_score
#2016-2019数据
# 指定文件夹路径
folder_path = ('Results')
data = pd.read_csv('DATA_Origin.csv')
Lon_Lat=data.iloc[:,0:2]

#print(Lon_Lat)


#Lon_Lat=np.array(Lon_Lat).reshape(,1)

# 获取所有Excel文件
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
csv_files=sorted(csv_files,key=lambda x:int(''.join(filter(str.isdigit,x))))
print(csv_files)
#空
combined_data_2016 = pd.DataFrame()
combined_data_2017 = pd.DataFrame()
combined_data_2018 = pd.DataFrame()
combined_data_2019 = pd.DataFrame()

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df_2016 = df.iloc[0,:]
    df_2016=np.array(df_2016)
    df_2016=df_2016.reshape(1,3)
    df_2016 = pd.DataFrame(df_2016,columns=['Ture','GLM-Attn+1D-CNN'])
    combined_data_2016 = pd.concat([combined_data_2016, df_2016], ignore_index=True)
    df_2017 = df.iloc[1,:]
    df_2017=np.array(df_2017)
    df_2017=df_2017.reshape(1,3)
    df_2017 = pd.DataFrame(df_2017,columns=['Ture','GLM-Attn+1D-CNN'])
    combined_data_2017 = pd.concat([combined_data_2017, df_2017], ignore_index=True)
    df_2018 = df.iloc[2,:]
    df_2018=np.array(df_2018)
    df_2018=df_2018.reshape(1,3)
    df_2018 = pd.DataFrame(df_2018,columns=['Ture','GLM-Attn+1D-CNN'])
    combined_data_2018 = pd.concat([combined_data_2018, df_2018], ignore_index=True)
    df_2019 = df.iloc[3,:]
    df_2019=np.array(df_2019)
    df_2019=df_2019.reshape(1,3)
    df_2019 = pd.DataFrame(df_2019,columns=['Ture','GLM-Attn+1D-CNN'])
    combined_data_2019 = pd.concat([combined_data_2019, df_2019], ignore_index=True)
# 保存合并后的结果
Yield_data1 = pd.concat([Lon_Lat,combined_data_2016],axis=1)
Yield_data1.to_csv("2016.csv", index=False)
Yield_data2 = pd.concat([Lon_Lat,combined_data_2017],axis=1)
Yield_data2.to_csv("2017.csv", index=False)
Yield_data3 = pd.concat([Lon_Lat,combined_data_2018],axis=1)
Yield_data3.to_csv("2018.csv", index=False)
Yield_data4 = pd.concat([Lon_Lat,combined_data_2019],axis=1)
Yield_data4.to_csv("2019.csv", index=False)
print("合并完成！")


#指标

df1 = pd.read_csv('2016.csv')
df2=pd.read_csv('2017.csv')
df3=pd.read_csv('2018.csv')
df4=pd.read_csv('2019.csv')
true_2016=df1.iloc[:,2]
predicted_2016=df1.iloc[:,3]
true_2017=df2.iloc[:,2]
predicted_2017=df2.iloc[:,3]
true_2018=df3.iloc[:,2]
predicted_2018=df3.iloc[:,3]
true_2019=df4.iloc[:,2]
predicted_2019=df4.iloc[:,3]

mse1 = mean_squared_error(true_2016, predicted_2016)
rmse1 = np.sqrt(mse1)
r21 = r2_score(true_2016, predicted_2016)

mse2 = mean_squared_error(true_2017, predicted_2017)
rmse2 = np.sqrt(mse2)
r22 = r2_score(true_2017, predicted_2017)

mse3 = mean_squared_error(true_2018, predicted_2018)
rmse3 = np.sqrt(mse3)
r23 = r2_score(true_2018, predicted_2018)

mse4 = mean_squared_error(true_2019, predicted_2019)
rmse4 = np.sqrt(mse4)
r24 = r2_score(true_2019,predicted_2019)
print(f'2016:mse{mse1},rmse{rmse1},r2{r21}')
print(f'2017:mse{mse2},rmse{rmse2},r2{r22}')
print(f'2018:mse{mse3},rmse{rmse3},r2{r23}')
print(f'2019:mse{mse4},rmse{rmse4},r2{r24}')