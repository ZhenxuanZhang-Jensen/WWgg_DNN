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

training_columns=["Diphoton_pt","Diphoton_eta","Diphoton_phi","scaled_leadphoton_pt","scaled_subleadphoton_pt","Diphoton_minID","Diphoton_maxID","LeadPhoton_eta","SubleadPhoton_eta","LeadPhoton_sigEoverE","SubleadPhoton_sigEoverE","Diphoton_mass_resolution","WW_E","WW_pt","WW_mass","WW_eta","WW_phi","W1_mass","W1_pt","W1_eta","W1_phi","W1_E","W2_mass","W2_pt","maxdR_gg_4jets","mindR_gg_4jets","maxdR_4jets","mindR_4jets","jet_1_pt","jet_1_E","jet_1_eta","jet_1_phi","jet_1_btagDeepFlavB","jet_2_pt","jet_2_E","jet_2_eta","jet_2_phi","jet_2_btagDeepFlavB","jet_3_pt","jet_3_E","jet_3_eta","jet_3_phi","jet_3_btagDeepFlavB","jet_4_pt","jet_4_E","jet_4_eta","jet_4_phi","jet_4_btagDeepFlavB","sum_two_max_bscore","costhetastar","nGoodAK4jets","costheta1","costheta2","Signal_Mass","Diphoton_dR"] 
mass_list = [250, 260, 270, 280, 300, 320, 350, 400, 450,550, 600, 650, 700, 750,800, 850, 900, 1000]

with open('/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/scaler.pkl','rb') as f:
    scaler = pickle.load(f)
data = pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/Data_2017.parquet')

data=data.query("Diphoton_minID>-0.7")

datadriven = pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/QCDreweighted.parquet')
datadriven=datadriven.query("Diphoton_minID>-0.7")
diphoton = pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/DiphotonJetsBox_reweighted.parquet')
diphoton = diphoton.query("Diphoton_minID>-0.7")

def load_trained_model(model_path):
    print('<load_trained_model> weights_path: ', model_path)
    model = load_model(model_path, compile=False)
    return model
model=load_trained_model("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/model_Relu_55_features_50epochs_pointsName.h5")
allsignal=[]
def plot_diphoton_mass(event1,):
    plt.hist
for file in mass_list:
    
    signalname='FH'+str(file)+'.parquet'
    # print(signal)
    signal= pd.read_parquet('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/'+signalname)
    signal=signal.query("Diphoton_minID>-0.7")
    signal['Signal_Mass']=file
    x_signal = signal[training_columns].values
    x_signal = scaler.transform(x_signal)
    pre_signal = model.predict(x_signal)
    signal['dnn_score'] = pre_signal
    allsignal.append(signal)
    signal.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat7/'+signalname)
    data['Signal_Mass']=file
    x_data = data[training_columns].values
    x_data = scaler.transform(x_data)
    pre_data = model.predict(x_data)
    data['dnn_score'] = pre_data
    data.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat7/data_'+str(file)+'.parquet')
    datadriven['Signal_Mass']=file
    diphoton['Signal_Mass']=file
    x_datadriven = datadriven[training_columns].values
    x_diphoton = diphoton[training_columns].values
    x_datadriven = scaler.transform(x_datadriven)
    x_diphoton = scaler.transform(x_diphoton)
    pre_datadriven = model.predict(x_datadriven)
    pre_diphoton = model.predict(x_diphoton)
    datadriven['dnn_score'] = pre_datadriven
    diphoton['dnn_score'] = pre_diphoton
    datadriven.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat7/QCD_'+str(file)+'.parquet')
    diphoton.to_parquet('/hpcfs/cms/cmsgpu/shaoweisong/output/cat7/DiphotonJetsbox_'+str(file)+'.parquet')

    fig, ax = plt.subplots()
    ax.hist([datadriven['dnn_score'], diphoton['dnn_score']], bins=40 ,alpha=0.8, label=['QCD', 'DiphotonJetsbox'], stacked=True, weights=[datadriven['weight_central'], diphoton['weight_central']], range=(0, 1))
    ax.hist(signal['dnn_score'], bins=40, alpha=0.8, label='10000*sig', color='red', edgecolor='red', weights=(10000 * signal['weight_central']), log=True,histtype="step")
    ax.hist(data['dnn_score'], bins=40, alpha=0.8, color='black', edgecolor='black', label='Data', histtype="step", log=True)
    ax.set_title('FH'+str(file)+' DNN score')
    ax.set_xlabel('DNN score')
    ax.legend()
    plt.savefig("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/DNNscore_"+str(file)+".png")
    plt.show()



