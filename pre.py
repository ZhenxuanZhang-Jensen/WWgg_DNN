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
import matplotlib.pyplot as plt
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

training_columns=["Diphoton_pt","scaled_leadphoton_pt","scaled_subleadphoton_pt","W1_mass","mindR_4jets","jet_1_pt","jet_2_pt","jet_2_E","jet_1_E","jet_1_btagDeepFlavB","jet_2_btagDeepFlavB","LeadPhoton_eta","SubleadPhoton_eta","W1_pt","jet_1_eta","jet_2_eta","LeadPhoton_sigEoverE","SubleadPhoton_sigEoverE","Diphoton_minID","Diphoton_maxID","Signal_Mass","Diphoton_dR"] 
# mass_list=[900, 260, 270, 280, 300, 320, 350, 400, 450,550, 600, 650, 700, 750, 800, 850, 900, 1000]
mass_list=[300]

with open('/hpcfs/cms/cmsgpu/shaoweisong/DNN/HHcat72jets/scaler.pkl','rb') as f:
    scaler = pickle.load(f)
data = pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/Data_2017.parquet')
# random_mass = random.choices(mass_list, k=len(data))
data['Signal_Mass']=300
data=data.query("Diphoton_minID>-0.7")

datadriven = pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/QCDreweighted.parquet')
# random_mass = random.choices(mass_list, k=len(datadriven))
datadriven['Signal_Mass']=300
datadriven=datadriven.query("Diphoton_minID>-0.7")
diphoton = pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/DiphotonJetsBox_reweighted.parquet')
# random_mass = random.choices(mass_list, k=len(diphoton))
diphoton['Signal_Mass']=300
diphoton = diphoton.query("Diphoton_minID>-0.7")

def load_trained_model(model_path):
    print('<load_trained_model> weights_path: ', model_path)
    model = load_model(model_path, compile=False)
    return model
model=load_trained_model("/hpcfs/cms/cmsgpu/shaoweisong/DNN/HHcat72jets/model_Relu_22_features_3epochs_pointsName.h5")
allsignal=[]
def plot_diphoton_mass(event1,):
    plt.hist
for file in mass_list:
    
    signalname='m'+str(file)+'.parquet'
    # print(signal)
    signal= pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/FH300.parquet')
    signal=signal.query("Diphoton_minID>-0.7")
    signal['Signal_Mass']=file
    x_signal = signal[training_columns].values
    x_signal = scaler.transform(x_signal)
    pre_signal = model.predict(x_signal)
    signal['dnn_score'] = pre_signal
    allsignal.append(signal)
    signal.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat7/'+signalname)
for i in mass_list:
    x_data = data[training_columns].values
    x_data = scaler.transform(x_data)
    pre_data = model.predict(x_data)
    data['dnn_score'] = pre_data
    data.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat7/data_'+str(i)+'.parquet')


x_datadriven = datadriven[training_columns].values
x_diphoton = diphoton[training_columns].values

x_datadriven = scaler.transform(x_datadriven)
x_diphoton = scaler.transform(x_diphoton)


pre_datadriven = model.predict(x_datadriven)
pre_diphoton = model.predict(x_diphoton)
datadriven['dnn_score'] = pre_datadriven
diphoton['dnn_score'] = pre_diphoton
fig, ax = plt.subplots()

ax.hist([datadriven['dnn_score'], diphoton['dnn_score']], bins=40 ,alpha=0.8, label=['QCD', 'DiphotonJetsbox'], stacked=True, weights=[datadriven['weight_central'], diphoton['weight_central']], range=(0, 1))
 
# ax.hist(datadriven['dnn_score'], bins=40, alpha=0.8, label='QCD', color='blue', edgecolor='blue', weights=datadriven['weight_central'], log=True)
# ax.hist(diphoton['dnn_score'], bins=40, alpha=0.8, label='DiphotonJetsbox', color='green', edgecolor='green', weights=diphoton['weight_central'], log=True)
ax.hist(signal['dnn_score'], bins=40, alpha=0.8, label='10000*sig', color='red', edgecolor='red', weights=(10000 * signal['weight_central']), log=True,histtype="step")

ax.hist(data['dnn_score'], bins=40, alpha=0.8, color='black', edgecolor='black', label='Data', histtype="step", log=True)

ax.set_title('DNN score')
ax.set_xlabel('DNN score')
ax.legend()
plt.savefig("/hpcfs/cms/cmsgpu/shaoweisong/DNN/HHcat72jets/DNNscore300.png")
plt.show()

datadriven.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat7/datadriven.parquet')
diphoton.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat7/diphoton.parquet')
