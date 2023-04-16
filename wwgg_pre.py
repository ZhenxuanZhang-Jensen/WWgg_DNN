import tensorflow as tf
import gc
import platform
import sys
import json
import xgboost as xgb
from functools import partial
from xgboost import plot_tree
from sklearn import metrics
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import numpy as np

import glob
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm.notebook import tqdm 
from IPython.display import display
from sklearn.metrics import mean_absolute_error as mae
import random
import os
from sklearn.preprocessing import StandardScaler
import pickle

training_columns=["scaled_leadphoton_pt","WW_pt","W1_mass","WW_mass","scaled_subleadphoton_pt","W2_mass","jet_1_pt","W1_pt","jet_4_pt","LeadPhoton_eta","SubleadPhoton_eta","jet_3_pt","jet_2_pt","W2_pt","nGoodAK4jets","Signal_Mass","Diphoton_minID","Diphoton_maxID","Diphoton_dR"] 
# mass_list=[250, 260, 270, 280, 300, 320, 350, 400, 450,550, 600, 650, 700, 750, 800, 850, 900, 1000]
mass_list=[600]

with open('/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat2/scaler.pkl','rb') as f:
    scaler = pickle.load(f)
data = pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat2/Data.parquet')
# random_mass = random.choices(mass_list, k=len(data))
data['Signal_Mass']=600
# data=data.query("minID>-0.7&leadjet_btagDeepFlavB>=0.0532&subleadjet_btagDeepFlavB>=0.0532")

datadriven = pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat2/DatadrivenQCD.parquet')
# random_mass = random.choices(mass_list, k=len(datadriven))
datadriven['Signal_Mass']=600
# datadriven=datadriven.query("minID>-0.7&leadjet_btagDeepFlavB>=0.0532&subleadjet_btagDeepFlavB>=0.0532")
diphoton = pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat2/DiphotonJetBox.parquet')
# random_mass = random.choices(mass_list, k=len(diphoton))
diphoton['Signal_Mass']=600
# diphoton = diphoton.query("minID>-0.7&leadjet_btagDeepFlavB>=0.0532&subleadjet_btagDeepFlavB>=0.0532")

def load_trained_model(model_path):
    print('<load_trained_model> weights_path: ', model_path)
    model = load_model(model_path, compile=False)
    return model
model=load_trained_model("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat2/model_Relu_19_features_200epochs_pointsName.h5")
allsignal=[]

for file in mass_list:
    
    signalname='m'+str(file)+'.parquet'
    # print(signal)
    signal= pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat2/m600.parquet')

    # signal= pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat2/'+signalname)
    # signal=signal.query("minID>-0.7&leadjet_btagDeepFlavB>=0.0532&subleadjet_btagDeepFlavB>=0.0532")
    signal['Signal_Mass']=file
    x_signal = signal[training_columns].values
    x_signal = scaler.transform(x_signal)
    pre_signal = model.predict(x_signal)
    signal['dnn_score'] = pre_signal
    signal['tran_dnn'] = np.exp(np.log(3)*signal.dnn_score*signal.dnn_score*signal.dnn_score)-2
    allsignal.append(signal)
    signal.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat2/'+signalname)
# allsignals=pd.concat(allsignal)
# allsignals.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat2/allsignals.parquet')
for i in mass_list:
    x_data = data[training_columns].values
    x_data = scaler.transform(x_data)
    pre_data = model.predict(x_data)
    data['dnn_score'] = pre_data
    data['tran_dnn'] = np.exp(np.log(3)*data.dnn_score*data.dnn_score*data.dnn_score)-2
    data.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat2/data_'+str(i)+'.parquet')


x_datadriven = datadriven[training_columns].values
x_diphoton = diphoton[training_columns].values

x_datadriven = scaler.transform(x_datadriven)
x_diphoton = scaler.transform(x_diphoton)


pre_datadriven = model.predict(x_datadriven)
pre_diphoton = model.predict(x_diphoton)



datadriven['dnn_score'] = pre_datadriven
# datadriven['tran_dnn'] = np.exp(np.log(3)*datadriven.dnn_score*datadriven.dnn_score*datadriven.dnn_score)-2
diphoton['dnn_score'] = pre_diphoton
# diphoton['tran_dnn'] = np.exp(np.log(3)*diphoton.dnn_score*diphoton.dnn_score*diphoton.dnn_score)-2

# #########
# data.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat2/data.parquet')
# for i in mass_list:
#     qcd=datadriven.query("Signal_Mass==%s"%(i))
#     dip=diphoton.query("Signal_Mass==%s"%(i))
#     qcd.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat2/datadriven_'+str(i)+'.parquet')
#     dip.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat2/diphoton_'+str(i)+'.parquet')
datadriven.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat2/datadriven.parquet')
diphoton.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat2/diphoton.parquet')
