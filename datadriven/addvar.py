from email.utils import decode_rfc2231
import awkward as ak
import numpy as np
import vector
vector.register_awkward()
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
    print(len(events))
    return events
def get_minmaxID(event):
    pho_pt=ak.concatenate([ak.unflatten(event.LeadPhoton_pt,counts=1),ak.unflatten(event.SubleadPhoton_pt,counts=1)],axis=1)
    pho_eta=ak.concatenate([ak.unflatten(event.LeadPhoton_eta,counts=1),ak.unflatten(event.SubleadPhoton_eta,counts=1)],axis=1)
    pho_phi=ak.concatenate([ak.unflatten(event.LeadPhoton_phi,counts=1),ak.unflatten(event.SubleadPhoton_phi,counts=1)],axis=1)
    pho_mass=ak.concatenate([ak.unflatten(event.LeadPhoton_mass,counts=1),ak.unflatten(event.SubleadPhoton_mass,counts=1)],axis=1)
    pho_ID=ak.concatenate([ak.unflatten(event.LeadPhoton_mvaID,counts=1),ak.unflatten(event.SubleadPhoton_mvaID,counts=1)],axis=1)
    pho_genPartFlav=ak.concatenate([ak.unflatten(event.LeadPhoton_genPartFlav,counts=1),ak.unflatten(event.SubleadPhoton_genPartFlav,counts=1)],axis=1)
    pho_genPartIdx=ak.concatenate([ak.unflatten(event.LeadPhoton_genPartIdx,counts=1),ak.unflatten(event.SubleadPhoton_genPartIdx,counts=1)],axis=1)
    photon = ak.zip({"pt":pho_pt,"eta":pho_eta,"phi":pho_phi,"mass":pho_mass,"ID":pho_ID,"genPartFlav":pho_genPartFlav,"genPartIdx":pho_genPartIdx})
    photon=photon[ak.argsort(photon.ID,ascending=False,axis=-1)]
    event['maxIDpt']=photon.pt[:,0]
    event['maxIDeta']=photon.eta[:,0]
    event['maxIDphi']=photon.phi[:,0]
    event['maxIDmass']=photon.mass[:,0]
    event['maxID_genPartFlav']=photon.genPartFlav[:,0]
    event['maxID_genPartIdx']=photon.genPartIdx[:,0]
    event['minIDpt']=photon.pt[:,1]
    event['minIDeta']=photon.eta[:,1]
    event['minIDphi']=photon.phi[:,1]
    event['minIDmass']=photon.mass[:,1]
    event['minID_genPartFlav']=photon.genPartFlav[:,1]
    event['minID_genPartIdx']=photon.genPartIdx[:,1]
    return event

inputfile=['/eos/user/s/shsong/HiggsDNA/fakephoton/UL17_GJet_Pt_20to40_DoubleEMEnriched_MGG_80toInf_2017/merged_nominal.parquet','/eos/user/s/shsong/HiggsDNA/fakephoton/UL17_GJet_Pt_20toInf_DoubleEMEnriched_MGG_40to80_2017/merged_nominal.parquet','/eos/user/s/shsong/HiggsDNA/fakephoton/UL17_GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_2017/merged_nominal.parquet']
outputfile=['/eos/user/s/shsong/HiggsDNA/parquet/bkg/Gjet_Pt20_40_2017.parquet','/eos/user/s/shsong/HiggsDNA/parquet/bkg/Gjet_Pt20_Inf_2017.parquet','/eos/user/s/shsong/HiggsDNA/parquet/bkg/Gjet_Pt40_Inf_2017.parquet']
i=0
for file in inputfile:
    event =load_parquet(file)
    print("Got event")
    event=get_minmaxID(event)
    print('Got get_minmaxID')

    ak.to_parquet(event, outputfile[i])
    if "Gjet" in outputfile[i]:
        parquet_to_root(outputfile[i],outputfile[i].replace("parquet","root"),treename="Gjet",verbose=False)

    if "DiPhotonJetsBox" in outputfile[i]:
        parquet_to_root(outputfile[i],outputfile[i].replace("parquet","root"),treename="DiPhotonJetsBox",verbose=False)
    i=i+1



