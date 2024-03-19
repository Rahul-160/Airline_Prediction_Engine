
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mlt
import pickle as pk
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc,roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import io
from sklearn.metrics import ConfusionMatrixDisplay



st.set_page_config(page_title='Airlines Delay Predictons',page_icon=' ✈️ ',
                   layout='centered')
st.markdown("<div style='background-color:#219C90; border-radius:50px;'><h1 style='text-align:center;color:white;'>Airlines Delay prediction  ✈️</h1></div",unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;color:white;'>don't delay your way</h3>",unsafe_allow_html=True)
st.image("airport-clipart-7.png",use_column_width=True)
data=pd.read_csv('Airlines.csv')
st.header('The Problem Statement')
st.write('-The objective of this preject is to analyze airline to gain insights into airplane delaying behavior and over all trends of the airplanes')

show_data=st.toggle(label='Show Data')
if show_data:
    st.dataframe(data,hide_index=True)
st.header('Link to the Resources')
st.markdown('-[Link to the Notebook](https://colab.research.google.com/drive/1j6L5DMM53mZ09nxiizpOrUVWhFSjrZNu?usp=sharing)')
st.markdown('-[Link to Dataset](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay)')
st.header('About the Dataset')
parms=st.selectbox(label='select any parameter',options=['shape','columns','description'],index=None)
if parms==None:
    st.write('please select an option...')
elif parms=='shape':
    st.write('shape of dataset:',data.shape)
elif parms=='description':
    st.write('description of dataset:')
    st.dataframe(data.describe(),use_container_width=True)
elif parms=='columns':
    st.write('columns of the airline dataset:')
    st.dataframe(data.columns,use_container_width=True)
st.header('Examine The Categorical Values')
cat_col=st.selectbox(label='select any Categorical Column',options=['Airline','AirportFrom','AirportTo'],index=None)
if cat_col is None:
    st.write('please select a column ...')
elif cat_col=='Airline':
    st.dataframe(data['Airline'].value_counts(),use_container_width=True)
elif cat_col=='AirportFrom':
    st.dataframe(data['AirportFrom'].value_counts(),use_container_width=True)
elif cat_col=='AirportTo':
    st.dataframe(data['AirportTo'].value_counts(),use_container_width=True)
st.header('Examine the Numerical column')
num_col=st.selectbox(label='select numerical column',options=['Flight','DayOfWeek','Time','Length'],index=None)
if num_col is None:
    st.write('plese select any numerical column...')
elif num_col=='Flight':
    st.write(f"count of evry unique values in **{num_col}** column:")
    st.dataframe(data[num_col].value_counts(),use_container_width=True)
elif num_col=='DayOfWeek':
    st.write(f"count of every unique values in **{num_col}** column:")
    st.dataframe(data[num_col].value_counts(),use_container_width=True)
elif num_col=='Time':
    st.write(f"count of every unique values in **{num_col}** column:")
    st.dataframe(data[num_col].value_counts(),use_container_width=True)
elif num_col=='Length':
    st.write(f"count of every unique values in **{num_col}** column:")
    st.dataframe(data[num_col].value_counts(),use_container_width=True)

st.header('Data Visualization')
data_col=st.selectbox(label='select any chart ',options=['Airline VS Counts',
                                                        'Day Of Week VS Crowd'],index=None)

if data_col=="Airline VS Counts":
    mlt.figure(figsize=(30, 30))
    mlt.subplot(4, 4, 1)
    sns.countplot(x='Airline', palette='Set2', data=data, order=data['Airline'].value_counts().index)
    st.pyplot()
    #st.set_option('deprecation.showPyplotGlobalUseWarning',False)
elif data_col=='Day Of Week VS Crowd':
    mlt.subplot(4,4,2)
    sns.countplot(x='DayOfWeek',palette='Set2',data=data,)
    st.pyplot()
elif data_col=='Airport VS Average Delay':
    mlt.subplot(4,4,3)
    mlt.scatter(x=data['AirportFrom'],y=data['Delay'])
    st.pyplot()
st.header('Show correlation')
uu=st.radio(label='press to see',options=['corr'],index=None)
if uu:
    corr=data.corr(numeric_only=True).round(2)
    mlt.figure(figsize=(15, 8))
    sns.heatmap(corr, annot=True)
    st.pyplot()

gra=st.selectbox(label='select delay vs feature graphs ',options=['airline vs delay','day vs delay'],index=True)
if gra=='airline vs delay':
    mlt.figure(figsize=(18, 20))
    mlt.subplot(2, 1, 1)
    sns.countplot(x='Airline', hue='Delay', palette='Set2', data=data)
    st.pyplot()
elif gra=='day vs delay':
    mlt.subplot(2, 1, 2)
    sns.countplot(x='DayOfWeek', hue='Delay', palette='Set2', data=data)
    st.pyplot()


st.header('Model Reports')
click_mo=st.button(label="click to generate model reports")
if click_mo:
    model1=pd.read_csv('model_report.csv')
    st.dataframe(model1)
st.write('**Confusion Matrix display**')
def oopen(filename,y):
    with open(filename,'rb') as f:
        y=pk.load(f)

y_test=pd.read_pickle('y_test')

def confusion(y):
    cm=confusion_matrix(y_test,y)
    mlt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt='g',cmap='Blues',cbar=False)
    mlt.xlabel('predicton label')
    mlt.ylabel('true labels')
    mlt.title('confusion matrix')
st.set_option('deprecation.showPyplotGlobalUse',False)
with open('y_kn', 'rb') as f:
    y_kn = pk.load(f)
with open('y_mn','rb') as f:
    y_mn=pk.load(f)
with open('y_rfc','rb') as f:
    y_rfc=pk.load(f)
with open('y_gn','rb') as f:
    y_gn=pk.load(f)
with open('y_xgbc','rb') as f:
    y_xgbc=pk.load(f)
with open('y_xg','rb') as f:
    y_xg=pk.load(f)
cm=st.selectbox(label='select the model',options=['knneighbors','multinomialNB','guassianNB','random forest','xgbclassifier','xgbrfclassifier'],index=None)
if cm=='knneighbors':
    confusion(y_kn)
    st.pyplot()
elif cm=='multinomialNB':
    confusion(y_mn)
    st.pyplot()
elif cm=='guassianNB':
    confusion(y_gn)
    st.pyplot()

elif cm=='random forest':
    confusion(y_rfc)
    st.pyplot()
elif cm=='xgbclassifier':
    confusion(y_xgbc)
    st.pyplot()
elif cm=='xgbrfclassifier':
    confusion(y_xg)
    st.pyplot()
st.write("**roc curves and auc")
st.write('The AUC-ROC curve, or Area Under the Receiver Operating Characteristic curve, is a graphical representation of the performance of a binary classification model at various classification problems. It is commonly used in machine learning to assess the ability of a model to distinguish between two classes.')
st.markdown('-[link for more information](https://www.geeksforgeeks.org/auc-roc-curve/#What%20Is%20The%20AUC-ROC%20curve?)')

def roooc(fpr_kn,tpr_kn):
    auc_kn=auc(fpr_kn,tpr_kn)
    mlt.figure()
    mlt.plot(fpr_kn,tpr_kn,color='darkorange',lw=2,label='auc rate (area under curve=%.2f)'%auc_kn)
    mlt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
    mlt.xlim([0.0,1.0])
    mlt.ylim([0.0,1.05])
    mlt.title('roc curve')
    mlt.xlabel('false positive rate')
    mlt.ylabel('true positie rate')
    mlt.legend(loc='lower right')

rau=st.selectbox(label='select any model..',options=['knneighbors','multinomialNB','guassianNB','random forest','xgbclassifier','xgbrfclassifier'],index=True)
if rau=='knneighbors':
    fpr_kn, tpr_kn, treshold = roc_curve(y_test, y_kn)
    roooc(fpr_kn,tpr_kn)
    st.pyplot()
elif rau=='multinomialNB':
    fpr_mn,tpr_mn,treshold_mn=roc_curve(y_test,y_mn)
    roooc(fpr_mn,tpr_mn)
    st.pyplot()
elif rau=='guassianNB':
    fpr_gn,tpr_gn,treshold_gn=roc_curve(y_test,y_gn)
    roooc(fpr_gn,tpr_gn)
    st.pyplot()
elif rau=='random forest':
    fpr_rfc,tpr_rfc,treshold_rfc=roc_curve(y_test,y_rfc)
    roooc(fpr_rfc,tpr_rfc)
    st.pyplot()
elif rau=='xgbclassifier':
    fpr_xgbc,tpr_xgbc,treshold_xgbc=roc_curve(y_test,y_xgbc)
    roooc(fpr_xgbc,tpr_xgbc)
    st.pyplot()
elif rau=='xgbrfclassifier':
    fpr_xg,tpr_xg,treshold_xgbc=roc_curve(y_test,y_xgbc)
    roooc(fpr_xg,tpr_xg)
    st.pyplot()
st.write('-As per roc_auc classification reports as so far xgbclassifier gives better results')
st.write('-xgbcclassifier also gives better recall')

st.header('Ovservations')
st.write('-As so for the model didt achive any vital progress in higher accuracy prediction.experimeted with knneighbors,multinomialNB,guassianNB,decision tree,random forest,xgbrfclassifier,xgbclassifier but none of them giving goos accuracy.however we have to take the model with higher recall accuracy')
st.write('xgbclassifier gives better recall accuracy so its our best model')
st.header('Predict Your Airline Delay')
airline=st.selectbox(label='select your ailine',options=['CO', 'US', 'AA', 'AS', 'DL', 'B6', 'HA', 'OO', '9E', 'OH', 'EV','XE', 'YV', 'UA', 'MQ', 'FL', 'F9', 'WN'])
flight=st.number_input(label='select your airline code(search internet)',step=1,value=0,format="%d")

fli=data['Flight'].values.tolist()
if flight in fli:
    st.write(f'you entered correct flight ID')
else:
    st.write('wrong airplane ID try again')
airportfrom=st.selectbox('select your jouney start from...',options=['SFO', 'PHX', 'LAX', 'ANC', 'LAS', 'SLC', 'DEN', 'ONT', 'FAI',
       'BQN', 'PSE', 'HNL', 'BIS', 'IYK', 'EWR', 'BOS', 'MKE', 'GFK',
       'OMA', 'GSO', 'LMT', 'SEA', 'MCO', 'TPA', 'DLH', 'MSP', 'FAR',
       'MFE', 'MSY', 'VPS', 'BWI', 'MAF', 'LWS', 'RST', 'ALB', 'DSM',
       'CHS', 'MSN', 'JAX', 'SAT', 'PNS', 'BHM', 'LIT', 'SAV', 'BNA',
       'ICT', 'ECP', 'DHN', 'MGM', 'CAE', 'PWM', 'ACV', 'EKO', 'PHL',
       'ATL', 'PDX', 'RIC', 'BTR', 'HRL', 'MYR', 'TUS', 'SBN', 'CAK',
       'TVC', 'CLE', 'ORD', 'DAY', 'MFR', 'BTV', 'TLH', 'TYS', 'DFW',
       'FLL', 'AUS', 'CHA', 'CMH', 'LRD', 'BRO', 'CRP', 'LAN', 'PVD',
       'FWA', 'JFK', 'LGA', 'OKC', 'PIT', 'PBI', 'ORF', 'DCA', 'AEX',
       'SYR', 'SHV', 'VLD', 'BDL', 'FAT', 'BZN', 'RDM', 'LFT', 'IPL',
       'EAU', 'ERI', 'BUF', 'IAH', 'MCI', 'AGS', 'ABI', 'GRR', 'LBB',
       'CLT', 'LEX', 'MBS', 'MOD', 'AMA', 'SGF', 'AZO', 'ABE', 'SWF',
       'BGM', 'AVP', 'FNT', 'GSP', 'ATW', 'ITH', 'TUL', 'COS', 'ELP',
       'ABQ', 'SMF', 'STL', 'IAD', 'DTW', 'RDU', 'RSW', 'OAK', 'ROC',
       'IND', 'CVG', 'MDW', 'SDF', 'ABY', 'TRI', 'XNA', 'ROA', 'MLI',
       'LYH', 'EVV', 'HPN', 'FAY', 'EWN', 'CSG', 'GPT', 'MLU', 'MOB',
       'OAJ', 'CHO', 'ILM', 'BMI', 'PHF', 'ACY', 'JAN', 'CID', 'GRK',
       'HOU', 'CRW', 'HTS', 'PSC', 'BOI', 'SBP', 'CLD', 'PSP', 'SBA',
       'MEM', 'MRY', 'GEG', 'RDD', 'PAH', 'CMX', 'SPI', 'EUG', 'CIC',
       'PIH', 'SGU', 'COD', 'MIA', 'MHT', 'GRB', 'FSD', 'SJU', 'AVL',
       'BFL', 'RAP', 'DRO', 'PIA', 'OGG', 'SIT', 'TXK', 'RNO', 'DAL',
       'SCE', 'MEI', 'MDT', 'FCA', 'SJC', 'KOA', 'PLN', 'SAN', 'GNV',
       'HLN', 'GJT', 'CPR', 'FSM', 'CMI', 'GTF', 'HDN', 'ITO', 'MTJ',
       'HSV', 'BTM', 'BIL', 'COU', 'MSO', 'SMX', 'TWF', 'ISP', 'GCC',
       'LIH', 'LNK', 'DAB', 'SNA', 'MQT', 'LGB', 'CWA', 'LSE', 'BUR',
       'ACT', 'MHK', 'MOT', 'IDA', 'SUN', 'GTR', 'MLB', 'SRQ', 'JAC',
       'ASE', 'LCH', 'JNU', 'ROW', 'BQK', 'YUM', 'FLG', 'EGE', 'GUC',
       'EYW', 'RKS', 'BGR', 'ELM', 'ADQ', 'OTZ', 'OTH', 'STT', 'KTN',
       'BET', 'SJT', 'CDC', 'CEC', 'SPS', 'SCC', 'STX', 'OME', 'MKG',
       'WRG', 'TYR', 'BRW', 'GGG', 'PSG', 'BKG', 'YAK', 'CLL', 'SAF',
       'CYS', 'LWB', 'CDV', 'FLO', 'BLI', 'DBQ', 'TOL', 'UTM', 'PIE',
       'ADK', 'ABR'])
airportto=st.selectbox('select where is your destination..',options=['IAH', 'CLT', 'DFW', 'SEA', 'MSP', 'DTW', 'ORD', 'ATL', 'PDX',
       'JFK', 'SLC', 'HNL', 'PHX', 'MCO', 'OGG', 'LAX', 'KOA', 'ITO',
       'SFO', 'MIA', 'IAD', 'SMF', 'PHL', 'LIH', 'DEN', 'LGA', 'MEM',
       'CVG', 'YUM', 'CWA', 'MKE', 'BQN', 'FAI', 'LAS', 'ANC', 'BOS',
       'LGB', 'FLL', 'SJU', 'EWR', 'DCA', 'BWI', 'RDU', 'MCI', 'TYS',
       'SAN', 'ONT', 'OAK', 'MDW', 'BNA', 'DAL', 'CLE', 'JAX', 'JNU',
       'RNO', 'ELP', 'SAT', 'OTZ', 'MBS', 'BDL', 'STL', 'HOU', 'AUS',
       'SNA', 'SJC', 'LIT', 'TUS', 'TUL', 'CMH', 'LAN', 'IND', 'AMA',
       'CRP', 'PIT', 'RKS', 'FWA', 'TPA', 'PBI', 'JAN', 'DSM', 'ADQ',
       'GRB', 'PVD', 'ABQ', 'SDF', 'RSW', 'MSY', 'BUR', 'BOI', 'TLH',
       'BHM', 'ACV', 'ORF', 'BET', 'KTN', 'RIC', 'SRQ', 'BTR', 'XNA',
       'MHT', 'GRR', 'SBN', 'SBA', 'ROA', 'CID', 'GPT', 'MFR', 'SGU',
       'HPN', 'OMA', 'OTH', 'GSP', 'LMT', 'BUF', 'MSN', 'BFL', 'CAE',
       'HRL', 'OKC', 'SYR', 'COS', 'BTV', 'CDC', 'SCC', 'DAY', 'SJT',
       'TVC', 'ROC', 'ISP', 'MRY', 'SBP', 'MLI', 'MOB', 'CIC', 'SAV',
       'FAT', 'EKO', 'GEG', 'ECP', 'LFT', 'SUN', 'HSV', 'SHV', 'CHA',
       'CAK', 'BZN', 'MAF', 'GSO', 'MDT', 'PHF', 'ICT', 'AZO', 'RAP',
       'CHS', 'CLD', 'MKG', 'VPS', 'PIH', 'ATW', 'AGS', 'PNS', 'BIL',
       'SPI', 'FAR', 'CPR', 'PIA', 'SPS', 'TWF', 'LBB', 'ALB', 'CEC',
       'DRO', 'GJT', 'GNV', 'RST', 'AVL', 'GRK', 'PSP', 'LEX', 'TRI',
       'SGF', 'FSM', 'RDD', 'OME', 'MFE', 'LSE', 'BMI', 'MYR', 'FAY',
       'FSD', 'EUG', 'MGM', 'EVV', 'MLB', 'FNT', 'STT', 'WRG', 'ABE',
       'BIS', 'MOT', 'MLU', 'GFK', 'RDM', 'COU', 'LRD', 'PSC', 'MOD',
       'PWM', 'ILM', 'ABY', 'CRW', 'TXK', 'BRO', 'BRW', 'EYW', 'DAB',
       'ROW', 'ABI', 'EAU', 'TYR', 'MSO', 'FLG', 'CSG', 'VLD', 'DHN',
       'OAJ', 'AEX', 'CHO', 'SAF', 'GGG', 'FCA', 'ASE', 'BKG', 'MHK',
       'LNK', 'MQT', 'YAK', 'GTR', 'SMX', 'SWF', 'ITH', 'AVP', 'ELM',
       'BGM', 'SIT', 'PSG', 'CYS', 'CLL', 'SCE', 'LWB', 'LCH', 'GCC',
       'IYK', 'LWS', 'COD', 'HLN', 'BQK', 'GTF', 'DLH', 'BTM', 'EGE',
       'IDA', 'JAC', 'HDN', 'MTJ', 'CMX', 'CMI', 'CDV', 'LYH', 'ACT',
       'STX', 'IPL', 'PAH', 'HTS', 'MEI', 'BLI', 'ERI', 'EWN', 'FLO',
       'ACY', 'DBQ', 'TOL', 'GUC', 'PLN', 'BGR', 'PSE', 'PIE', 'UTM',
       'ADK', 'ABR'])
dayofweek=st.selectbox(label="select your day of travel",options=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])

if dayofweek=='Sunday':
    dayofweek=1
elif dayofweek=='Monday':
    dayofweek=2
elif dayofweek=='Tuesday':
    dayofweek=3
elif dayofweek=='Wednesday':
    dayofweek=4
elif dayofweek=='Thursday':
    dayofweek=5
elif dayofweek=='Friday':
    dayofweek=6
elif dayofweek=='Saturday':
    dayofweek=7
times=st.number_input(label='Time of your flight in the format HHMM railway time',step=1,value=0,format="%d")
if times>0 and times<9999 :
    st.write('your flight time is ',times)
else:
    st.write('please enter time in right format')

length=st.slider(label='select your travel distance',min_value=0,max_value=655)


with open('labelairline.pk','rb') as f:
    label_airline=pk.load(f)
with open('labelairportfrom','rb') as f:
    label_airportfrom=pk.load(f)
with open('labelairportto','rb') as f:
    label_airportto=pk.load(f)
with open('xg_rightone.pk','rb') as f:
    model_xgbc=pk.load(f)
with open('scaler.pk','rb') as f:
    scalar_minmax=pk.load(f)
airline=[airline]
airline=label_airline.transform(airline)
airportto=[airportto]
airportto=label_airportto.transform(airportto)
airportfrom=[airportfrom]
airportfrom=label_airportfrom.transform(airportfrom)

parmeters=[airline[0],flight,airportfrom[0],airportto[0],dayofweek,times,length]

par=pd.DataFrame(parmeters)

par=par.transpose()
par.reset_index(drop=True,inplace=True)



st.header('Predict Your Airline Whether Delay OR Not')
st.write('-This airline delay prediction is the combination of data science expertise and domain knowledge to dvolop effective models that help ailines better manage their operations'
         'and improve the passenger experience')

kk=st.button(label='PREDICT')
if kk:
    parmeters = scalar_minmax.transform(par)
    pred=model_xgbc.predict(parmeters)
    if pred==1:
        st.write('your air line will be delayed')
    if pred==0:
        st.write('be quick!your airline sharp on time')










