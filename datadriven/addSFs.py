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
    events=events[events.category==7]
    return events
def add_sale_factor(event,sclae_factor):
    event['weight_central']=sclae_factor*event.weight_central
    return event
# inputfile=['/eos/user/s/shsong/HHWWgg/parquet/cat7_2jet/DatadrivenQCD.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7_2jet/DiPhotonJetsBox_MGG_80toInf_20171.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7_2jet/DiPhotonJetsBox_MGG_80toInf_20172.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7_2jet/DiPhotonJetsBox_M40_80_2017.parquet']
# outputfile_FH=['/eos/user/s/shsong/HHWWgg/parquet/cat7_2jet/QCDreweighted.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7_2jet/DiphotonJetsBox_reweighted.parquet']
# inputfile=['/eos/user/s/shsong/HHWWgg/parquet/cat7_3jet/DatadrivenQCD.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7_3jet/DiPhotonJetsBox_MGG_80toInf_20171.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7_3jet/DiPhotonJetsBox_MGG_80toInf_20172.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7_3jet/DiPhotonJetsBox_M40_80_2017.parquet']
# outputfile_FH=['/eos/user/s/shsong/HHWWgg/parquet/cat7_3jet/QCDreweighted.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7_3jet/DiphotonJetsBox_reweighted.parquet']
# inputfile=['/eos/user/s/shsong/HHWWgg/parquet/cat7_4jet/DatadrivenQCD.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7_4jet/DiPhotonJetsBox_MGG_80toInf_20171.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7_4jet/DiPhotonJetsBox_MGG_80toInf_20172.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7_4jet/DiPhotonJetsBox_M40_80_2017.parquet']
# outputfile_FH=['/eos/user/s/shsong/HHWWgg/parquet/cat7_4jet/QCDreweighted.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7_4jet/DiphotonJetsBox_reweighted.parquet']
inputfile=['/eos/user/s/shsong/HHWWgg/parquet/cat7/DatadrivenQCD.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiPhotonJetsBox_MGG_80toInf_20171.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiPhotonJetsBox_MGG_80toInf_20172.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiPhotonJetsBox_M40_80_2017.parquet']
outputfile_FH=['/eos/user/s/shsong/HHWWgg/parquet/cat7/QCDreweighted.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiphotonJetsBox_reweighted.parquet']
eventQCD=load_parquet(inputfile[0])
eventDiphoton1=load_parquet(inputfile[1])
eventDiphoton2=load_parquet(inputfile[2])
eventDiphoton3=load_parquet(inputfile[3])
eventDiphoton=ak.concatenate([eventDiphoton1,eventDiphoton2,eventDiphoton3])
# QCD_reweighted = add_sale_factor(eventQCD,1.08637) #2jet
# GG_reweighted = add_sale_factor(eventDiphoton,1.3149) #2jet
# QCD_reweighted = add_sale_factor(eventQCD,1.03184) #3jet
# GG_reweighted = add_sale_factor(eventDiphoton,1.25069) #3jet
# QCD_reweighted = add_sale_factor(eventQCD,1.01856) #4jet
# GG_reweighted = add_sale_factor(eventDiphoton,1.17111) #4jet
QCD_reweighted = add_sale_factor(eventQCD,1.06674) 
GG_reweighted = add_sale_factor(eventDiphoton,1.27355) 
ak.to_parquet(QCD_reweighted, outputfile_FH[0])
parquet_to_root(outputfile_FH[0],outputfile_FH[0].replace("parquet","root"),treename="cat7",verbose=False)
ak.to_parquet(GG_reweighted, outputfile_FH[1])
parquet_to_root(outputfile_FH[1],outputfile_FH[1].replace("parquet","root"),treename="cat7",verbose=False)
