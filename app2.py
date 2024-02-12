import pandas as pd #pengolahan data tabel seperti select field
import numpy as np #pengolahan angka, matriks, vektor
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import ndiffs
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Uas Kelompok 1",
                 layout= "wide")

data = pd.read_excel('\\Users\\20200\\Desktop\\uas\\DATA PRODUK EDIFIER TRAINING.xlsx')

st.title( "UAS Kelompok Prediksi Penjualan")
st.title( "TABEL PRODUK EDIFIER")

def convert_df(df):
   return data.to_csv(index=False).encode('utf-8')

st.dataframe(data)

st.sidebar.header("Filter:")
nama_produk = st.sidebar.multiselect(
    "Select the nama_produk:",
     options=data["nama_produk"].unique(),
     default=[]
    )

tanggal_keluar = st.sidebar.multiselect(
    "Select the tanggal masuk:",
     options=data["tanggal_keluar"].unique(),
     default=[]
    )

kuantitas_terjual = st.sidebar.multiselect(
    "Select the kuantitas terjual:",
     options=data["kuantitas_terjual"].unique(),
     default=[]
    )

df_selection=data.query(
  "nama_produk== @nama_produk & tanggal_keluar== @tanggal_keluar & kuantitas_terjual == @kuantitas_terjual"
)
st.dataframe(df_selection)

data_path='\\Users\\20200\\Desktop\\uas\\DATA PRODUK EDIFIER TRAINING.xlsx'

data = pd.read_excel(data_path)

df_train=pd.DataFrame(data)

"""# Data Training"""

var_input=df_train[['nama_produk','kuantitas','harga_produk_satuan','harga_jual_satuan','pendapatan_keseluruhan']] #Variabel atribut/feature yang akan digunakan
var_target=df_train[['kuantitas_terjual']] #Variabel target/label

import warnings
import itertools
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')


matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

df_train['nama_produk'].value_counts()

penjualan = df_train.loc[:len(data)]

penjualan['tanggal_keluar'].min(), penjualan['tanggal_keluar'].max()

cols = ['kode_produk','nama_produk','tipe_produk','kuantitas','tanggal_masuk','harga_produk_satuan','harga_jual_satuan','pendapatan_keseluruhan']
penjualan.drop(cols, axis=1, inplace=True)

penjualan = penjualan.sort_values('tanggal_keluar') #Dataframe akan kita sort berdasarkan tanggal
penjualan.isnull().sum() #Lalu selanjutnya akan menjumlahkan jumlah kolom yang null, bisa dilihat hasilnya sudah 0

st.title("Data Training")
penjualan
penjualan = penjualan.groupby('tanggal_keluar')['kuantitas_terjual'].sum().reset_index()
penjualan = penjualan.set_index('tanggal_keluar')

y = penjualan['kuantitas_terjual'].resample('W').mean()
y = y.dropna()

st.title("Total Penjualan Perminggu")
y['2022':]

y.plot(figsize=(15, 6))
plt.ylabel('kuantitas_terjual (kuantitas terjual)',fontsize=18)
plt.xlabel('tanggal_terjual',fontsize=18)
plt.title('tanggal vs kuantitas terjual')
plt.show() #Menampilkan plot qty per-bulan

st.title("Total Penjualan Perhari")
penjualan

from pylab import rcParams #
import statsmodels.api as sm
rcParams['figure.figsize'] = 12, 5
decomposition = sm.tsa.seasonal_decompose(y, model='additive', period=4)
fig = decomposition.plot()
plt.show()

p = d = q = range(0, 2) #Rumus dari algoritma ARIMA Seasonal
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq: #Fit Model
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 0, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary()) #Hasil untuk model AR dan MA serta ARIMA

st.title( "ARIMA")
pred = results.get_prediction(start=pd.to_datetime('2022-11-06'), dynamic=False)
pred_ci = pred.conf_int()
fig, ax = plt.subplots(figsize=(25, 7))
y['2022':].plot(label='observed', ax=ax)
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=0.2)
ax.set_xlabel('tanggal_keluar')
ax.set_ylabel('kuantitas_terjual')
plt.legend()
st.pyplot(fig)


y_forecasted = pred.predicted_mean
y_truth = y['2022-01-03':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error is {}'.format(round(mse, 2))) #Menampilkan MSE
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2))) #Menampilkan RMSE


st.title( "SARIMA")
pred_uc = results.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
fig1, ax = plt.subplots(figsize=(25, 7))
data['kuantitas_terjual'].plot(label='observed', ax=ax)
pred_uc.predicted_mean.plot(ax=ax, label='forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('tanggal_keluar')
ax.set_ylabel('kuantitas_terjual')
plt.legend()
st.pyplot(fig1)

pred_ci
pred_ci.to_csv('predictions.csv')