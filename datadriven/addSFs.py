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
inputfile=['/eos/user/s/shsong/HHWWgg/parquet/cat7/DatadrivenQCD.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiPhotonJetsBox_MGG_80toInf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiPhotonJetsBox_M40_80_2017.parquet']
outputfile_FH=['/eos/user/s/shsong/HHWWgg/parquet/cat7/QCDreweighted.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiphotonJetsBox_reweighted.parquet']
eventQCD=load_parquet(inputfile[0])
eventDiphoton1=load_parquet(inputfile[1])
eventDiphoton2=load_parquet(inputfile[2])
eventDiphoton=ak.concatenate([eventDiphoton1,eventDiphoton2])
QCD_reweighted = add_sale_factor(eventQCD,1.06415)
GG_reweighted = add_sale_factor(eventDiphoton,1.27936)
ak.to_parquet(QCD_reweighted, outputfile_FH[0])
parquet_to_root(outputfile_FH[0],outputfile_FH[0].replace("parquet","root"),treename="cat7",verbose=False)
ak.to_parquet(GG_reweighted, outputfile_FH[1])
parquet_to_root(outputfile_FH[1],outputfile_FH[1].replace("parquet","root"),treename="cat7",verbose=False)
