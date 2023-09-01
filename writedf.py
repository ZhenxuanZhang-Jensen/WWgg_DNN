
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
import random
import glob
import pandas as pd
import sys
from tqdm.notebook import tqdm 
from IPython.display import display
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler
import pickle
import re

def scheduler(epoch):
  if epoch < 10:
    return 0.0001
  else:
    return 0.0000001

def clearVar(var):
    del var
    gc.collect()

def readDF(f,processID,isBkg:bool):
    df=pd.read_parquet(f)
    df=df.query("Diphoton_minID>-0.7")
   
    if(isBkg):
        mass = [250, 260, 270, 280, 300, 320, 350, 400, 450,550, 600, 650, 700, 750, 800, 850, 900, 1000]
        random_mass = random.choices(mass, k=len(df))
        df["target"]=0
        df["weight"]=df["weight_central"]
        df['Signal_Mass']=random_mass
    else:
        mx = re.search("FH(\d+)", f)
        df["target"]=1
        df['Signal_Mass']=int(mx.group(1))
        df["weight"]=df["weight_central"]/df["Diphoton_mass_resolution"]

    df["processID"]=processID
    # print("file name is:",f.split("/")[-1],"number of events:",df.shape[0],"weighted events:",df.weight.sum())
    return df
def getClassWeight(df):
    #normalize signal to backgroud
    # background target == 0; sig target ==1
    bkg_yields=df.groupby("target")["weight"].sum()[0]
    sig_yields=df.groupby("target")["weight"].sum()[1]
    N_bkg=df.query("target==0").shape[0]
    N=bkg_yields
    norm_bkg=N/bkg_yields
    norm_sig=N/sig_yields
    # norm=sig_yields/bkg_yields
    df["class_weight"]=df["target"].map({0:norm_bkg, 1:norm_sig})
    df["new_weight"]= df["class_weight"]* df["weight"]

    print("original wighted events:",bkg_yields,sig_yields)
    print("new weight:",df.groupby("target")["new_weight"].sum())
    return df,norm_bkg,norm_sig

def getMX(filename):
    mx = re.search("FH(\d+)", filename)
    print(filename)
    print(mx)
    if mx:
        mx = mx.group(1)
    return "Signal_M"+str(mx)+"_point"


sigFiles=glob.glob("/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/FH*parquet")
# sigFiles=glob.glob("/hpcfs/cms/cmsgpu/shaoweisong/input/cat7_2jet/FH*parquet")
AllSample="gghh"
signals=[]
for f in sigFiles:
    processID=getMX(f)
    df=readDF(f,processID,False)
    print("load",processID)
    print("concatenate %s"%f)

    signals.append(df)
signal=pd.concat(signals)
signal['type']=3
QCD=readDF('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/QCDreweighted.parquet',"BKG_QCD",True)
QCD['type']=1
Dipho=readDF('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/DiphotonJetsBox_reweighted.parquet',"BKG_diphoton",True)
Dipho['type']=2

data_set=pd.concat([QCD,Dipho,signal])
# data_set=data_set.fillna(-999)

#normalize signal to background
data_set,norm_bkg,norm_sig=getClassWeight(data_set)
print("norm factor:",norm_bkg,norm_sig)



train_model=True
epochs=200
gridSearch=False
useXGB=False
compute_importance=True

# Create a LearningRateScheduler callback that uses the scheduler function
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


data_set.to_csv('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/dnn_massresolutionreweighted.csv')
# data_set.to_csv('/hpcfs/cms/cmsgpu/shaoweisong/DNN/dnn.csv')


