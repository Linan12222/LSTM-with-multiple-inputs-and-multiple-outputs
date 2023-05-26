import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
plt.rcParams['axes.unicode_minus']=False       #显示负号

from keras.models import Model
from keras.layers import Dense,Input,CuDNNLSTM,LSTM
from keras.optimizers import  Adam

from keras.utils.vis_utils import plot_model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#data_=pd.read_excel('./2018风电数据_modify.xlsx')
data_=pd.read_excel('./data_proccessed.xlsx')
look_back=4

#归一化
scaler=MinMaxScaler((0,1))
data_scale=scaler.fit_transform(data_)

#窗口函数
def split_sequence(sequence, look_back):
    X, y = [], []
    for i in range(len(sequence)):
        end_element_index = i + look_back
        if end_element_index > len(sequence) - 1: # 序列中最后一个元素的索引
            break
        sequence_x, sequence_y = sequence[i:end_element_index], sequence[end_element_index] # 取最后一个元素作为预测值y
        X.append(sequence_x)
        if len(sequence_y.shape)>1:
            y.append(sequence_y)
        else:
            y.append(sequence_y)

    return np.array(X), np.array(y)

#划分数据集
x,y=split_sequence(data_scale,look_back=look_back)
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


#模型建立
main_in=Input(shape=(look_back,X_train.shape[-1]))
temp=CuDNNLSTM(units=64, return_sequences=False)(main_in)#无法运行就使用下一行
# temp=LSTM(units=64, return_sequences=False,activation='relu')(main_in)
temp=Dense(20,activation='relu')(temp)
temp=Dense(32,activation='relu')(temp)
main_out=Dense(int(y_train.shape[-1]))(temp)

model=Model(inputs=[main_in],outputs=[main_out])
#模型可视化
plot_model(model,show_shapes=True)
model.compile(loss=['mse'],optimizer=Adam(0.001),metrics=['mae'])

#模型训练和可视化
history=model.fit(X_train,y_train,epochs=100,batch_size=1000,verbose=2,validation_split=1/9)#训练集为0.9，验证集为训练集的1/9，也就是整体的0.1
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

#---------------------------模型测试----------------------------#
preds=model(X_test)
#反归一化
predictions=scaler.inverse_transform(preds)
#预测结果
df_pre=pd.DataFrame(predictions)
#保存预测结果到excel
df_pre.to_excel('./predictions.xlsx')
#真实值
df_real=pd.DataFrame(scaler.inverse_transform(y_test))

#----------------------------------指标计算------------------------#

mae = (df_pre - df_real).abs().mean()
mse = ((df_pre - df_real) ** 2).mean()
r2 = df_pre.apply(lambda col: r2_score(col, df_real[col.name]))
#每一列的MAE、MSE、R2
print('MAE:\n', mae)
print('MSE:\n', mse)
print('R2:\n', r2)

#画图
for col in df_real.columns:
    plt.figure(figsize=(30,8),dpi=100)
    plt.plot(df_real[col], label='df_real',marker='.')
    plt.plot(df_pre[col], label='df_pre',marker='*')
    plt.legend()
    plt.title(col)
    plt.show()
