import awkward as ak
import numpy as np
import pandas as pd
import sys
from parquet_to_root import parquet_to_root
import sys
from random import choice
from math import *
import concurrent.futures

def load_parquet(fname):
    print("loading events from %s" % fname)
    events=ak.from_parquet(fname)
    # events=events[events.category==2]
    events=events[events.category==4]
    # events=events[events.category==7]
    return events
def add_sale_factor(event,sclae_factor):
    event['weight_central']=sclae_factor*event.weight_central
    return event
inputfile=['/eos/user/s/shsong/HHWWgg/parquet/cat4/DatadrivenQCD.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat4/DiPhotonJetsBox_MGG_80toInf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat4/DiPhotonJetsBox_M40_80_2017.parquet']
outputfile_FH=['/eos/user/s/shsong/HHWWgg/parquet/cat4/QCDreweighted.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat4/DiphotonJetsBox_reweighted.parquet']
# inputfile=['/eos/user/s/shsong/HHWWgg/parquet/cat7/DatadrivenQCD.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiPhotonJetsBox_MGG_80toInf_20171.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiPhotonJetsBox_MGG_80toInf_20172.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiPhotonJetsBox_M40_80_2017.parquet']
# outputfile_FH=['/eos/user/s/shsong/HHWWgg/parquet/cat7/QCDreweighted.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiphotonJetsBox_reweighted.parquet']
# inputfile=['/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/DatadrivenQCD.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/DiPhotonJetsBox_MGG_80toInf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/DiPhotonJetsBox_M40_80_2017.parquet']
# outputfile_SL=['/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/QCDreweighted.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/DiphotonJetsBox_reweighted.parquet']
eventQCD=load_parquet(inputfile[0])
print(len(eventQCD))
eventDiphoton1=load_parquet(inputfile[1])
eventDiphoton2=load_parquet(inputfile[2])
# eventDiphoton2=load_parquet(inputfile[3])
eventDiphoton=ak.concatenate([eventDiphoton1,eventDiphoton2])
# eventDiphoton=ak.concatenate([eventDiphoton1,eventDiphoton2,eventDiphoton3])
# QCD_reweighted = add_sale_factor(eventQCD,1.06674) #cat7
# GG_reweighted = add_sale_factor(eventDiphoton,1.27355)  #cat7
# QCD_reweighted = add_sale_factor(eventQCD,0.784433) #cat2
# GG_reweighted = add_sale_factor(eventDiphoton,7.00168)  #cat2
QCD_reweighted = add_sale_factor(eventQCD,0.520145) #cat4
GG_reweighted = add_sale_factor(eventDiphoton,1.31187)  #cat4
# QCD_reweighted['LeadPhoton_E_over_mass']=QCD_reweighted['LeadPhoton_E']/QCD_reweighted['Diphoton_mass']
# GG_reweighted['LeadPhoton_E_over_mass']=GG_reweighted['LeadPhoton_E']/GG_reweighted['Diphoton_mass']
# QCD_reweighted['SubleadPhoton_E_over_mass']=QCD_reweighted['SubleadPhoton_E']/QCD_reweighted['Diphoton_mass']
# GG_reweighted['SubleadPhoton_E_over_mass']=GG_reweighted['SubleadPhoton_E']/GG_reweighted['Diphoton_mass']

# ak.to_parquet(QCD_reweighted, outputfile_SL[0])
# parquet_to_root(outputfile_SL[0],outputfile_SL[0].replace("parquet","root"),treename="cat2",verbose=False)
# ak.to_parquet(GG_reweighted, outputfile_SL[1])
# parquet_to_root(outputfile_SL[1],outputfile_SL[1].replace("parquet","root"),treename="cat2",verbose=False)
ak.to_parquet(QCD_reweighted, outputfile_FH[0])
print(len(QCD_reweighted))
print(len(GG_reweighted))
parquet_to_root(outputfile_FH[0],outputfile_FH[0].replace("parquet","root"),treename="cat4",verbose=False)
ak.to_parquet(GG_reweighted, outputfile_FH[1])
parquet_to_root(outputfile_FH[1],outputfile_FH[1].replace("parquet","root"),treename="cat4",verbose=False)
