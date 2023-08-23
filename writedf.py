import sys
import json
import numpy as np
import random
import glob
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re


def readDF(f,processID,isBkg:bool):
    df=pd.read_parquet(f)
    df=df.query("Diphoton_minID>-0.7")
   
    if(isBkg):
        mass = [250, 300, 400, 450,550, 650, 750, 900]
        random_mass = random.choices(mass, k=len(df))
        df["target"]=0
        df["weight"]=df["weight_central"]
        df['Signal_Mass']=random_mass
    else:
        mx = re.search("FH(\d+)", f)
        df["target"]=1
        df['Signal_Mass']=int(mx.group(1))
        df["weight"]=df["weight_central"]


    df["processID"]=processID
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

    if mx:
        mx = mx.group(1)
    return "Signal_M"+str(mx)+"_point"


sigFiles=glob.glob("/eos/user/s/shsong/HHWWgg/parquet/cat7/FH*.parquet")
print(sigFiles)
AllSample="gghh"
signals=[]
for f in sigFiles:
    processID=getMX(f)
    df=readDF(f,processID,False)
    print("load",processID)
    signals.append(df)
signal=pd.concat(signals)

QCD=readDF('/eos/user/s/shsong/HHWWgg/parquet/cat7/QCDreweighted.parquet',"BKG_QCD",True)
Dipho=readDF('/eos/user/s/shsong/HHWWgg/parquet/cat7/DiphotonJetsBox_reweighted.parquet',"BKG_diphoton",True)
data_set=pd.concat([QCD,Dipho,signal])
data_set=data_set.fillna(-1)

#normalize signal to background
data_set,norm_bkg,norm_sig=getClassWeight(data_set)
print("norm factor:",norm_bkg,norm_sig)


data_set.to_csv('/eos/user/s/shsong/HHWWgg/parquet/cat7/DNN/dnn.csv')
# data_set.to_csv('/hpcfs/cms/cmsgpu/shaoweisong/DNN/dnn.csv')
