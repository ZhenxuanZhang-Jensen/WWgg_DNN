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
    eventsSL=events[events.category==2]
    eventsFH=events[events.category==7]
    return eventsSL,eventsFH
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
def RenameDfQCD(df):
     df=df.rename({"Diphoton_mass":"Diphoton_mass", 
                       "LeadPhoton_mvaID":"Leading_Photon_MVA_old",
                        "SubleadPhoton_mvaID":"Subleading_Photon_MVA_old"
                       }, axis='columns')
     print(df.columns)
     return df
def select_jet(event):
    event['SelectedJet_pt']=event.jet_1_pt
    event['SelectedJet_eta']=event.jet_1_eta
    event['SelectedJet_phi']=event.jet_1_phi
    event['SelectedJet_mass']=event.jet_1_mass
    event['SelectedJet_jetId']=event.jet_1_jetId
    event['SelectedJet_puId']=event.jet_1_puId
    event['SelectedJet_btagDeepFlavB']=event.jet_1_btagDeepFlavB
    return event
def calclulate_W_info(event):
    leadjet=vector.obj(pt=event.jet_1_pt,eta=event.jet_1_eta,phi=event.jet_1_phi,mass=event.jet_1_mass)
    event['jet_1_E']=np.nan_to_num(leadjet.E, nan=-1)

    subleadjet=vector.obj(pt=event.jet_2_pt,eta=event.jet_2_eta,phi=event.jet_2_phi,mass=event.jet_2_mass)
    event['jet_2_E']=np.nan_to_num(subleadjet.E, nan=-1)
    dummy = ak.zeros_like(leadjet.pt)
    dummy = ak.ones_like(ak.fill_none(dummy, 0))
    event['jet_2_E']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_2_E'])
    event['jet_2_pt']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_2_pt'])
    event['jet_2_eta']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_2_eta'])
    event['jet_2_phi']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_2_phi'])
    event['jet_2_mass']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_2_mass'])
    W1=leadjet+subleadjet
    event['W1_pt']=np.nan_to_num(W1.pt,nan=-1)
    event['W1_eta']=np.nan_to_num(W1.eta,nan=-1)
    event['W1_phi']=np.nan_to_num(W1.phi,nan=-1)
    event['W1_mass']=np.nan_to_num(W1.mass,nan=-1)
    event['W1_E']=np.nan_to_num(W1.E, nan=-1)
    event['W1_pt']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['W1_pt'])
    event['W1_eta']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['W1_eta'])
    event['W1_phi']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['W1_phi'])
    event['W1_mass']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['W1_mass'])
    event['W1_E']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['W1_E'])

    subsubleadjet=vector.obj(pt=event.jet_3_pt,eta=event.jet_3_eta,phi=event.jet_3_phi,mass=event.jet_3_mass)
    event['jet_3_E']=np.nan_to_num(subsubleadjet.E, nan=-999.0)
    event['jet_3_E']=ak.where(subsubleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_3_E'])
    event['jet_3_pt']=ak.where(subsubleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_3_pt'])
    event['jet_3_eta']=ak.where(subsubleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_3_eta'])
    event['jet_3_phi']=ak.where(subsubleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_3_phi'])
    event['jet_3_mass']=ak.where(subsubleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_3_mass'])
    subsubsubleadjet=vector.obj(pt=event.jet_4_pt,eta=event.jet_4_eta,phi=event.jet_4_phi,mass=event.jet_4_mass)
    event['jet_4_E']=np.nan_to_num(subsubsubleadjet.E, nan=-999.0)
    event['jet_4_E']=ak.where(subsubsubleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_4_E'])
    event['jet_4_pt']=ak.where(subsubsubleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_4_pt'])
    event['jet_4_eta']=ak.where(subsubsubleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_4_eta'])
    event['jet_4_phi']=ak.where(subsubsubleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_4_phi'])
    event['jet_4_mass']=ak.where(subsubsubleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_4_mass'])
    
    event['dphi_j1j2']=abs(event.jet_1_phi-event.jet_2_phi)
    event['dphi_j1j2']=ak.where(((event.jet_1_pt<0)|(event.jet_2_pt<0)),(ak.ones_like(dummy)*(-1)),event['dphi_j1j2'])
    event['dphi_j1j3']=abs(event.jet_1_phi-event.jet_3_phi)
    event['dphi_j1j3']=ak.where(((event.jet_1_pt<0)|(event.jet_3_pt<0)),(ak.ones_like(dummy)*(-1)),event['dphi_j1j3'])
    event['dphi_j1j4']=abs(event.jet_1_phi-event.jet_4_phi)
    event['dphi_j1j4']=ak.where(((event.jet_1_pt<0)|(event.jet_4_pt<0)),(ak.ones_like(dummy)*(-1)),event['dphi_j1j4'])
    event['dphi_j2j3']=abs(event.jet_2_phi-event.jet_3_phi)
    event['dphi_j2j3']=ak.where(((event.jet_2_pt<0)|(event.jet_3_pt<0)),(ak.ones_like(dummy)*(-1)),event['dphi_j2j3'])
    event['dphi_j2j4']=abs(event.jet_2_phi-event.jet_4_phi)
    event['dphi_j2j4']=ak.where(((event.jet_2_pt<0)|(event.jet_4_pt<0)),(ak.ones_like(dummy)*(-1)),event['dphi_j2j4'])
    event['dphi_j3j4']=abs(event.jet_3_phi-event.jet_4_phi)
    event['dphi_j3j4']=ak.where(((event.jet_3_pt<0)|(event.jet_4_pt<0)),(ak.ones_like(dummy)*(-1)),event['dphi_j3j4'])
    W2=subsubleadjet+subsubsubleadjet
    event['W2_pt']=W2.pt
    event['W2_eta']=W2.eta
    event['W2_phi']=W2.phi
    event['W2_mass']=W2.mass
    event['W2_E']=np.nan_to_num(W2.E, nan=-1)
    
    event['W2_E']=ak.where(((subsubleadjet.pt<0)|(subsubsubleadjet.pt<0)),(ak.ones_like(dummy)*(-1)),event['W2_E'])
    event['W2_pt']=ak.where(((subsubleadjet.pt<0)|(subsubsubleadjet.pt<0)),(ak.ones_like(dummy)*(-1)),event['W2_pt'])
    event['W2_eta']=ak.where(((subsubleadjet.pt<0)|(subsubsubleadjet.pt<0)),(ak.ones_like(dummy)*(-1)),event['W2_eta'])
    event['W2_phi']=ak.where(((subsubleadjet.pt<0)|(subsubsubleadjet.pt<0)),(ak.ones_like(dummy)*(-1)),event['W2_phi'])
    event['W2_mass']=ak.where(((subsubleadjet.pt<0)|(subsubsubleadjet.pt<0)),(ak.ones_like(dummy)*(-1)),event['W2_mass'])
        
    WW=W1+W2
    event['WW_pt']=WW.pt
    event['WW_eta']=WW.eta
    event['WW_phi']=WW.phi
    event['WW_mass']=WW.mass
    event['WW_E']=WW.E
    
    event['WW_E']=ak.where(((event['W1_pt']<0)|(event['W2_pt']<0)),(ak.ones_like(dummy)*(-1)),event['WW_E'])
    event['WW_pt']=ak.where(((event['W1_pt']<0)|(event['W2_pt']<0)),(ak.ones_like(dummy)*(-1)),event['WW_pt'])
    event['WW_eta']=ak.where(((event['W1_pt']<0)|(event['W2_pt']<0)),(ak.ones_like(dummy)*(-1)),event['WW_eta'])
    event['WW_phi']=ak.where(((event['W1_pt']<0)|(event['W2_pt']<0)),(ak.ones_like(dummy)*(-1)),event['WW_phi'])
    event['WW_mass']=ak.where(((event['W1_pt']<0)|(event['W2_pt']<0)),(ak.ones_like(dummy)*(-1)),event['WW_mass'])

    return event
def scaled_diphoton_info(event):
    event['scaled_leadphoton_pt']=event['LeadPhoton_pt']/event['Diphoton_mass']
    event['scaled_subleadphoton_pt']=event['SubleadPhoton_pt']/event['Diphoton_mass']
    leadphoton=vector.obj(pt=event.LeadPhoton_pt,eta=event.LeadPhoton_eta,phi=event.LeadPhoton_phi,mass=event.LeadPhoton_mass)
    subleadphoton=vector.obj(pt=event.SubleadPhoton_pt,eta=event.SubleadPhoton_eta,phi=event.SubleadPhoton_phi,mass=event.SubleadPhoton_mass)
    event['LeadPhoton_E_over_mass']=leadphoton.E/event.Diphoton_mass
    event['SubleadPhoton_E_over_mass']=subleadphoton.E/event.Diphoton_mass
#     event['Diphoton_mass']=event['CMS_hgg_mass']
    return event
def calculate_bscore(event):
    nparr=np.array(event.jet_1_btagDeepFlavB)
    new_array = nparr.astype('float64')
    event['jet_1_btagDeepFlavB']=new_array
    event['jet_1_btagDeepFlavB']
    nparr=np.array(event.jet_2_btagDeepFlavB)
    new_array = nparr.astype('float64')
    event['jet_2_btagDeepFlavB']=new_array
    event['jet_2_btagDeepFlavB']
    nparr=np.array(event.jet_3_btagDeepFlavB)
    new_array = nparr.astype('float64')
    event['jet_3_btagDeepFlavB']=new_array
    event['jet_3_btagDeepFlavB']
    nparr=np.array(event.jet_4_btagDeepFlavB)
    new_array = nparr.astype('float64')
    event['jet_4_btagDeepFlavB']=new_array
    event['jet_4_btagDeepFlavB']
    nparr=np.array(event.jet_5_btagDeepFlavB)
    new_array = nparr.astype('float64')
    event['jet_5_btagDeepFlavB']=new_array
    event['jet_5_btagDeepFlavB']
    nparr=np.array(event.jet_6_btagDeepFlavB)
    new_array = nparr.astype('float64')
    event['jet_6_btagDeepFlavB']=new_array
    event['jet_6_btagDeepFlavB']
    nparr=np.array(event.jet_7_btagDeepFlavB)
    new_array = nparr.astype('float64')
    event['jet_7_btagDeepFlavB']=new_array
    event['jet_7_btagDeepFlavB']
    jet1btagDeepFlavB=np.reshape(ak.to_numpy(event['jet_1_btagDeepFlavB']),(len(event),1))
    jet2btagDeepFlavB=np.reshape(ak.to_numpy(event['jet_2_btagDeepFlavB']),(len(event),1))
    jet3btagDeepFlavB=np.reshape(ak.to_numpy(event['jet_3_btagDeepFlavB']),(len(event),1))
    jet4btagDeepFlavB=np.reshape(ak.to_numpy(event['jet_4_btagDeepFlavB']),(len(event),1))
    jet5btagDeepFlavB=np.reshape(ak.to_numpy(event['jet_5_btagDeepFlavB']),(len(event),1))
    jet6btagDeepFlavB=np.reshape(ak.to_numpy(event['jet_6_btagDeepFlavB']),(len(event),1))
    jet7btagDeepFlavB=np.reshape(ak.to_numpy(event['jet_7_btagDeepFlavB']),(len(event),1))
    jetbtagDeepFlavB=ak.concatenate((jet1btagDeepFlavB,jet2btagDeepFlavB,jet3btagDeepFlavB,jet4btagDeepFlavB,jet5btagDeepFlavB,jet6btagDeepFlavB,jet7btagDeepFlavB),axis=1)
    sortjet=ak.sort(jetbtagDeepFlavB,axis=-1,ascending=False)
    sum_two_max_bscore=sortjet[:,1]+sortjet[:,0]
    event['sum_two_max_bscore']=sum_two_max_bscore
    return event
def getCosThetaStar_CS(objects1,objects2):
    objects1=ak.zip({
    "pt":event['Diphoton_pt'],
    "eta":event['Diphoton_eta'],
    "phi":event['Diphoton_phi'],
    "mass":event['Diphoton_mass']},with_name="Momentum4D")
    objects2=ak.zip({
    "pt":event['WW_pt'],
    "eta":event['WW_eta'],
    "phi":event['WW_phi'],
    "mass":event['WW_mass']},with_name="Momentum4D")

    HH=objects1+objects2    
    boost_vec=vector.obj(px=-(HH.px)/HH.E,py=-(HH.py)/HH.E,pz=-(HH.pz)/HH.E)
    obj1=objects1.boost(boost_vec)    
    p=np.sqrt(obj1.px*obj1.px+obj1.py*obj1.py+obj1.pz*obj1.pz)

    return ak.flatten(abs(obj1.pz/p))
def E_resolution(event):
    leadphoton=vector.obj(pt=event.LeadPhoton_pt,eta=event.LeadPhoton_eta,phi=event.LeadPhoton_phi,mass=event.LeadPhoton_mass)
    event['LeadPhoton_E']=leadphoton.E
    # leadpho_px=event['LeadPhoton_pt']*np.cos(event['LeadPhoton_phi'])
    # leadpho_py=event['LeadPhoton_pt']*np.sin(event['LeadPhoton_phi'])
    # leadpho_pz=event['LeadPhoton_pt']*np.sinh(event['LeadPhoton_eta'])
    # event['LeadPhoton_E']=np.sqrt(np.array(leadpho_px)*np.array(leadpho_px)+np.array(leadpho_py)*np.array(leadpho_py)+np.array(leadpho_pz)*np.array(leadpho_pz)-event['LeadPhoton_mass']*event['LeadPhoton_mass'])
    event['LeadPhoton_sigEoverE']=event['LeadPhoton_energyErr']/event['LeadPhoton_E']
    subleadphoton=vector.obj(pt=event.SubleadPhoton_pt,eta=event.SubleadPhoton_eta,phi=event.SubleadPhoton_phi,mass=event.SubleadPhoton_mass)
    event['SubleadPhoton_E']=subleadphoton.E
    # subleadpho_px=event['SubleadPhoton_pt']*np.cos(event['SubleadPhoton_phi'])
    # subleadpho_py=event['SubleadPhoton_pt']*np.sin(event['SubleadPhoton_phi'])
    # subleadpho_pz=event['SubleadPhoton_pt']*np.sinh(event['SubleadPhoton_eta'])
    # event['SubleadPhoton_E']=np.sqrt(np.array(subleadpho_px)*np.array(subleadpho_px)+np.array(subleadpho_py)*np.array(subleadpho_py)+np.array(subleadpho_pz)*np.array(subleadpho_pz)-event['SubleadPhoton_mass']*event['SubleadPhoton_mass'])
    event['SubleadPhoton_sigEoverE']=event['SubleadPhoton_energyErr']/event['SubleadPhoton_E']
    event['Diphoton_mass_resolution']=0.5*np.sqrt(event['LeadPhoton_sigEoverE']**2+event['SubleadPhoton_sigEoverE']**2)
    return event
def getCosThetaStar(event):
    objects1=ak.zip({
    "pt":event['Diphoton_pt'],
    "eta":event['Diphoton_eta'],
    "phi":event['Diphoton_phi'],
    "mass":event['Diphoton_mass']},with_name="Momentum4D")
    objects2=ak.zip({
    "pt":event['WW_pt'],
    "eta":event['WW_eta'],
    "phi":event['WW_phi'],
    "mass":event['WW_mass']},with_name="Momentum4D")
    HH=objects1+objects2 #obj could be HH condidate    
    # vx=HH.px / (HH.mass * HH.beta)
    # vy=HH.py / (HH.mass * HH.beta)
    # vz=HH.pz / (HH.mass * HH.beta)
    # event['costhetastar']=vz
    p= np.sqrt(HH.px**2 + HH.py**2 + HH.pz**2)
    event['costhetastar']=HH.pz/p
    dummy = ak.zeros_like(objects1.pt)
    dummy = ak.ones_like(ak.fill_none(dummy, 0))
    event['costhetastar']=ak.where((objects2.pt<0),(ak.ones_like(dummy)*(-1)),event['costhetastar'])

    return event
def calculate_dR_gg_4jets(event):
    jet_pt=ak.concatenate([np.reshape(ak.to_numpy(event['jet_1_pt']),(len(event),1)),np.reshape(ak.to_numpy(event['jet_2_pt']),(len(event),1)),np.reshape(ak.to_numpy(event['jet_3_pt']),(len(event),1)),np.reshape(ak.to_numpy(event['jet_4_pt']),(len(event),1))],axis=1)
    jet_eta=ak.concatenate([np.reshape(ak.to_numpy(event['jet_1_eta']),(len(event),1)),np.reshape(ak.to_numpy(event['jet_2_eta']),(len(event),1)),np.reshape(ak.to_numpy(event['jet_3_eta']),(len(event),1)),np.reshape(ak.to_numpy(event['jet_4_eta']),(len(event),1))],axis=1)
    jet_phi=ak.concatenate([np.reshape(ak.to_numpy(event['jet_1_phi']),(len(event),1)),np.reshape(ak.to_numpy(event['jet_2_phi']),(len(event),1)),np.reshape(ak.to_numpy(event['jet_3_phi']),(len(event),1)),np.reshape(ak.to_numpy(event['jet_4_phi']),(len(event),1))],axis=1)
    jet_mass=ak.concatenate([np.reshape(ak.to_numpy(event['jet_1_mass']),(len(event),1)),np.reshape(ak.to_numpy(event['jet_2_mass']),(len(event),1)),np.reshape(ak.to_numpy(event['jet_3_mass']),(len(event),1)),np.reshape(ak.to_numpy(event['jet_4_mass']),(len(event),1))],axis=1)
    jet=ak.zip({"pt":jet_pt,"eta":jet_eta,"phi":jet_phi,"mass":jet_mass},with_name="Momentum4D")
    Leadphoton=ak.zip({"pt":event['LeadPhoton_pt'],"eta":event['LeadPhoton_eta'],"phi":event['LeadPhoton_phi'],"mass":event['LeadPhoton_mass']},with_name="Momentum4D")
    SubleadPhoton=ak.zip({"pt":event['SubleadPhoton_pt'],"eta":event['SubleadPhoton_eta'],"phi":event['SubleadPhoton_phi'],"mass":event['SubleadPhoton_mass']},with_name="Momentum4D")
    Jet = ak.unflatten(jet, counts = 1) # shape [n_events, n_obj, 1]
    lead_photon = ak.unflatten(Leadphoton, counts = 1)  # shape [n_events, 1, n_obj]
    sublead_photon = ak.unflatten(SubleadPhoton, counts = 1)  # shape [n_events, 1, n_obj]
    lead_pho_jet_dr=ak.flatten(Jet.deltaR(lead_photon))
    sublead_pho_jet_dr=ak.flatten(Jet.deltaR(sublead_photon))
    diphoton_jetdR=ak.concatenate([lead_pho_jet_dr,sublead_pho_jet_dr],axis=1)
    dummy = ak.zeros_like(Leadphoton.pt)
    dummy = ak.ones_like(ak.fill_none(dummy, 0))
    diphoton_jetdR=ak.where((diphoton_jetdR>10),dummy,diphoton_jetdR)
    event['maxdR_gg_4jets']=diphoton_jetdR[ak.argsort(diphoton_jetdR)][:,7]
    jjggdRnoNAN=ak.fill_none(ak.pad_none(diphoton_jetdR[diphoton_jetdR!=-1],1),-1)
    event['mindR_gg_4jets']=jjggdRnoNAN[ak.argsort(jjggdRnoNAN)][:,0]
    return event
def calculate_dR_4jets(event):
    jet1=ak.zip({"pt":event.jet_1_pt,"eta":event.jet_1_eta,"phi":event.jet_1_phi,"phi":event.jet_1_phi},with_name="Momentum4D")
    jet2=ak.zip({"pt":event.jet_2_pt,"eta":event.jet_2_eta,"phi":event.jet_2_phi,"phi":event.jet_2_phi},with_name="Momentum4D")
    jet3=ak.zip({"pt":event.jet_3_pt,"eta":event.jet_3_eta,"phi":event.jet_3_phi,"phi":event.jet_3_phi},with_name="Momentum4D")
    jet4=ak.zip({"pt":event.jet_4_pt,"eta":event.jet_4_eta,"phi":event.jet_4_phi,"phi":event.jet_4_phi},with_name="Momentum4D")
    dummy = ak.zeros_like(jet1.pt)
    dummy = ak.ones_like(ak.fill_none(dummy, 0))
    j1j2dr=ak.unflatten(ak.where(((jet1.pt<0)|(jet2.pt<0)),(ak.ones_like(dummy)*(-1)),jet1.deltaR(jet2)),counts=1)
    j1j3dr=ak.unflatten(ak.where(((jet1.pt<0)|(jet3.pt<0)),(ak.ones_like(dummy)*(-1)),jet1.deltaR(jet3)),counts=1)
    j1j4dr=ak.unflatten(ak.where(((jet1.pt<0)|(jet4.pt<0)),(ak.ones_like(dummy)*(-1)),jet1.deltaR(jet4)),counts=1)
    j2j3dr=ak.unflatten(ak.where(((jet2.pt<0)|(jet3.pt<0)),(ak.ones_like(dummy)*(-1)),jet2.deltaR(jet3)),counts=1)
    j2j4dr=ak.unflatten(ak.where(((jet2.pt<0)|(jet4.pt<0)),(ak.ones_like(dummy)*(-1)),jet2.deltaR(jet4)),counts=1)
    j3j4dr=ak.unflatten(ak.where(((jet3.pt<0)|(jet4.pt<0)),(ak.ones_like(dummy)*(-1)),jet3.deltaR(jet4)),counts=1)

    jjdR=ak.concatenate([j1j2dr,j1j3dr,j1j4dr,j2j3dr,j2j4dr,j3j4dr],axis=1)
    event['maxdR_4jets']=jjdR[ak.argsort(jjdR)][:,5]
    jjdRnoNAN=ak.fill_none(ak.pad_none(jjdR[jjdR!=-1],1),-1)
    event['mindR_4jets']=jjdRnoNAN[ak.argsort(jjdRnoNAN)][:,0]
    return event
def add_sale_factor(event,sclae_factor):
    event['weight_central']=sclae_factor*event.weight_central
    return event
def momentum_tensor(list_of_jets_lorentzvectors_):
    M_xy = np.array([[0.,0.],[0.,0.]])
    for v_ in list_of_jets_lorentzvectors_:
        #Transverse momentum tensor (symmetric matrix)
        M_xy += np.array([
        [v_.px*v_.px,v_.px*v_.py],
        [v_.px*v_.py,v_.py*v_.py]]
        )
        eigvals, eigvecs = np.linalg.eig(M_xy)
    eigvals.sort()
    return eigvals, eigvecs
def sphericity(eigvals):
    # Definition: http://sro.sussex.ac.uk/id/eprint/44644/1/art%253A10.1007%252FJHEP06%25282010%2529038.pdf
    spher_ = 2*eigvals[0] / (eigvals[1]+eigvals[0])
    return spher_
def costheta1(event):
    W1=ak.zip({
    "pt":event['W1_pt'],
    "eta":event['W1_eta'],
    "phi":event['W1_phi'],
    "mass":event['W1_mass']},with_name="Momentum4D")
    H1=ak.zip({
    "pt":event['WW_pt'],
    "eta":event['WW_eta'],
    "phi":event['WW_phi'],
    "mass":event['WW_mass']},with_name="Momentum4D")
    H2=ak.zip({
    "pt":event['Diphoton_pt'],
    "eta":event['Diphoton_eta'],
    "phi":event['Diphoton_phi'],
    "mass":event['Diphoton_mass']},with_name="Momentum4D")
    HH=H1+H2
    # H1 rest frame
    boost_vec=vector.obj(px=H1.px / H1.E,py=H1.py / H1.E,pz=H1.pz / H1.E)
#     boost_vec=vector.obj(px=HH.px / (HH.E*HH.mass),py=HH.py / (HH.E*HH.mass),pz=HH.pz / (HH.E*HH.mass))
    W1_rest=W1.boost(boost_vec)
    HH_rest=HH.boost(boost_vec)
    # calculate W1 HH momentum magnitude
    p1 = np.sqrt(W1_rest.px**2 + W1_rest.py**2 + W1_rest.pz**2)
    p2 = np.sqrt(HH_rest.px**2 + HH_rest.py**2 + HH_rest.pz**2)
    # calculate  W1 unit vector and HH unit vector 
    ux1 = W1_rest.px / p1
    uy1 = W1_rest.py / p1
    uz1 = W1_rest.pz / p1
    ux2 = HH_rest.px / p2
    uy2 = HH_rest.py / p2
    uz2 = HH_rest.pz / p2
    # The dot product of two unit vectors is equal to cos theta
    cos_theta = ux1 * ux2 + uy1 * uy2 + uz1 * uz2
    event['costheta1']=cos_theta
    dummy = ak.zeros_like(W1.pt)
    dummy = ak.ones_like(ak.fill_none(dummy, 0))
    event['costheta1']=ak.where(((H1.pt<0)|(H2.pt<0)|(W1.pt<0)),(ak.ones_like(dummy)*(-1)),event['costheta1'])

    return event
def costheta2(event):
    W2=ak.zip({
    "pt":event['W2_pt'],
    "eta":event['W2_eta'],
    "phi":event['W2_phi'],
    "mass":event['W2_mass']},with_name="Momentum4D")
    H1=ak.zip({
    "pt":event['WW_pt'],
    "eta":event['WW_eta'],
    "phi":event['WW_phi'],
    "mass":event['WW_mass']},with_name="Momentum4D")
    H2=ak.zip({
    "pt":event['Diphoton_pt'],
    "eta":event['Diphoton_eta'],
    "phi":event['Diphoton_phi'],
    "mass":event['Diphoton_mass']},with_name="Momentum4D")
    HH=H1+H2
    # H1 rest frame
    boost_vec=vector.obj(px=H2.px / H2.E,py=H2.py / H2.E,pz=H2.pz / H2.E)
#     boost_vec=vector.obj(px=HH.px / (HH.E*HH.mass),py=HH.py / (HH.E*HH.mass),pz=HH.pz / (HH.E*HH.mass))
    W2_rest=W2.boost(boost_vec)
    HH_rest=HH.boost(boost_vec)
    # calculate W1 HH momentum magnitude
    p1 = np.sqrt(W2_rest.px**2 + W2_rest.py**2 + W2_rest.pz**2)
    p2 = np.sqrt(HH_rest.px**2 + HH_rest.py**2 + HH_rest.pz**2)
    # calculate  W1 unit vector and HH unit vector 
    ux1 = W2_rest.px / p1
    uy1 = W2_rest.py / p1
    uz1 = W2_rest.pz / p1
    ux2 = HH_rest.px / p2
    uy2 = HH_rest.py / p2
    uz2 = HH_rest.pz / p2
    # The dot product of two unit vectors is equal to cos theta
    cos_theta = ux1 * ux2 + uy1 * uy2 + uz1 * uz2
    event['costheta2']=cos_theta
    dummy = ak.zeros_like(W2.pt)
    dummy = ak.ones_like(ak.fill_none(dummy, 0))
    event['costheta2']=ak.where(((H1.pt<0)|(H2.pt<0)|(W2.pt<0)),(ak.ones_like(dummy)*(-1)),event['costheta2'])
    return event
def calculate_sphericity(event):
    sphericity_list=[]
    jet_pt=ak.concatenate([ak.unflatten(event.jet_1_pt,counts=1),ak.unflatten(event.jet_2_pt,counts=1),ak.unflatten(event.jet_3_pt,counts=1),ak.unflatten(event.jet_4_pt,counts=1)],axis=1)
    jet_eta=ak.concatenate([ak.unflatten(event.jet_1_eta,counts=1),ak.unflatten(event.jet_2_eta,counts=1),ak.unflatten(event.jet_3_eta,counts=1),ak.unflatten(event.jet_4_eta,counts=1)],axis=1)
    jet_phi=ak.concatenate([ak.unflatten(event.jet_1_phi,counts=1),ak.unflatten(event.jet_2_phi,counts=1),ak.unflatten(event.jet_3_phi,counts=1),ak.unflatten(event.jet_4_phi,counts=1)],axis=1)
    jet_mass=ak.concatenate([ak.unflatten(event.jet_1_mass,counts=1),ak.unflatten(event.jet_2_mass,counts=1),ak.unflatten(event.jet_3_mass,counts=1),ak.unflatten(event.jet_4_mass,counts=1)],axis=1)
    jet_btag=ak.concatenate([ak.unflatten(event.jet_1_btagDeepB,counts=1),ak.unflatten(event.jet_2_btagDeepB,counts=1),ak.unflatten(event.jet_3_btagDeepB,counts=1),ak.unflatten(event.jet_4_btagDeepB,counts=1)],axis=1)
    select_jet=ak.zip({"pt":jet_pt,"eta":jet_eta,"phi":jet_phi,"mass":jet_mass,"btag":jet_btag},with_name="Momentum4D")
    for j in range(len(event)):
        eigvals, eigvecs = momentum_tensor([select_jet[j][0],select_jet[j][1],select_jet[j][2],select_jet[j][3]])
        spher_ = sphericity(eigvals)
        sphericity_list.append(spher_)    
    event['sphericity']=np.array(sphericity_list)
    return event
def calculate_sphericity_new(event):
    sphericity_list=[]
    jet_pt=ak.concatenate([ak.unflatten(event.jet_1_pt,counts=1),ak.unflatten(event.jet_2_pt,counts=1),ak.unflatten(event.jet_3_pt,counts=1),ak.unflatten(event.jet_4_pt,counts=1)],axis=1)
    jet_eta=ak.concatenate([ak.unflatten(event.jet_1_eta,counts=1),ak.unflatten(event.jet_2_eta,counts=1),ak.unflatten(event.jet_3_eta,counts=1),ak.unflatten(event.jet_4_eta,counts=1)],axis=1)
    jet_phi=ak.concatenate([ak.unflatten(event.jet_1_phi,counts=1),ak.unflatten(event.jet_2_phi,counts=1),ak.unflatten(event.jet_3_phi,counts=1),ak.unflatten(event.jet_4_phi,counts=1)],axis=1)
    jet_mass=ak.concatenate([ak.unflatten(event.jet_1_mass,counts=1),ak.unflatten(event.jet_2_mass,counts=1),ak.unflatten(event.jet_3_mass,counts=1),ak.unflatten(event.jet_4_mass,counts=1)],axis=1)
    jet_btag=ak.concatenate([ak.unflatten(event.jet_1_btagDeepB,counts=1),ak.unflatten(event.jet_2_btagDeepB,counts=1),ak.unflatten(event.jet_3_btagDeepB,counts=1),ak.unflatten(event.jet_4_btagDeepB,counts=1)],axis=1)
    select_jet=ak.zip({"pt":jet_pt,"eta":jet_eta,"phi":jet_phi,"mass":jet_mass,"btag":jet_btag},with_name="Momentum4D")
    
    def calculate_sphericity(jet):
        eigvals, eigvecs = momentum_tensor([jet[0], jet[1], jet[2], jet[3]])
        spher_ = sphericity(eigvals)
        return spher_
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        sphericity_list = list(executor.map(calculate_sphericity, select_jet))

    event['sphericity'] = np.array(sphericity_list)
    return event
def calculate_photon_info(event):
    event['scaled_leadphoton_pt']=event['LeadPhoton_pt']/event['Diphoton_mass']
    event['scaled_subleadphoton_pt']=event['SubleadPhoton_pt']/event['Diphoton_mass']

    return event
def calculate_Wjj(event):
    leadjet=vector.obj(pt=event.jet_1_pt,eta=event.jet_1_eta,phi=event.jet_1_phi,mass=event.jet_1_mass)
    event['jet_1_E']=np.nan_to_num(leadjet.E, nan=-1)
    subleadjet=vector.obj(pt=event.jet_2_pt,eta=event.jet_2_eta,phi=event.jet_2_phi,mass=event.jet_2_mass)
    event['jet_2_E']=np.nan_to_num(subleadjet.E, nan=-1)
    dummy = ak.zeros_like(leadjet.pt)
    dummy = ak.ones_like(ak.fill_none(dummy, 0))
    event['jet_2_E']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_2_E'])
    event['jet_2_pt']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_2_pt'])
    event['jet_2_eta']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_2_eta'])
    event['jet_2_phi']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_2_phi'])
    event['jet_2_mass']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['jet_2_mass'])
    
    W1=leadjet+subleadjet
    event['W1_pt']=np.nan_to_num(W1.pt,nan=-1)
    event['W1_eta']=np.nan_to_num(W1.eta,nan=-1)
    event['W1_phi']=np.nan_to_num(W1.phi,nan=-1)
    event['W1_mass']=np.nan_to_num(W1.mass,nan=-1)
    event['W1_E']=np.nan_to_num(W1.E, nan=-1)
    event['W1_pt']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['W1_pt'])
    event['W1_eta']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['W1_eta'])
    event['W1_phi']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['W1_phi'])
    event['W1_mass']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['W1_mass'])
    event['W1_E']=ak.where(subleadjet.pt<0,(ak.ones_like(dummy)*(-1)),event['W1_E'])
    return event
def lepton(event):
    dummy = ak.zeros_like(event.PuppiMET_pt)
    dummy = ak.ones_like(ak.fill_none(dummy, 0))
    lepton_pt=ak.concatenate([ak.unflatten(event.electron_iso_pt,counts=1),ak.unflatten(event.muon_iso_pt,counts=1)],axis=1)
    lepton_eta=ak.concatenate([ak.unflatten(event.electron_iso_eta,counts=1),ak.unflatten(event.muon_iso_eta,counts=1)],axis=1)
    lepton_phi=ak.concatenate([ak.unflatten(event.electron_iso_phi,counts=1),ak.unflatten(event.muon_iso_phi,counts=1)],axis=1)
    lepton_mass=ak.concatenate([ak.unflatten(event.electron_iso_mass,counts=1),ak.unflatten(event.muon_iso_mass,counts=1)],axis=1)
    lepton=vector.obj(pt=ak.flatten(lepton_pt[lepton_pt!=-999],axis=1),eta=ak.flatten(lepton_eta[lepton_eta!=-999],axis=1),phi=ak.flatten(lepton_phi[lepton_phi!=-999],axis=1),mass=ak.flatten(lepton_mass[lepton_mass!=-999],axis=1))
    event['lepton_iso_pt']=lepton.pt
    event['lepton_iso_eta']=lepton.eta
    event['lepton_iso_phi']=lepton.phi
    event['lepton_iso_pt']=lepton.mass
    event['lepton_iso_E']=lepton.E
    lepton_Et=np.sqrt(lepton.mass*lepton.mass+lepton.pt*lepton.pt)
    # lepton_Et=(lepton.E/np.sqrt(lepton.E*lepton.E-lepton.mass*lepton.mass))*lepton.pt
    Mt=np.sqrt(lepton.mass*lepton.mass+2*(lepton_Et*event.MET_pt-lepton.pt*event.MET_pt))
    event['Mt_lepMET']=Mt
    event_dphi=(event.jet_1_phi-event.PuppiMET_phi)
    event_negative_jetphi=ak.fill_none((event_dphi.mask[event_dphi<-np.pi]+2*np.pi),np.nan)
    event_negative_jetphi=ak.fill_none((event_dphi.mask[event_dphi<-np.pi]+2*np.pi),np.nan)
    event_positive_jetphi=ak.fill_none((-event_dphi.mask[event_dphi>np.pi]+2*np.pi),np.nan)
    event_middphi=ak.fill_none((event_dphi.mask[(event_dphi<=np.pi)&(event_dphi>=-np.pi)]),np.nan)
    jet_new_phi = np.where(~np.isnan(event_positive_jetphi), event_positive_jetphi, event_negative_jetphi)
    event['dphi_jet1_PuppiMET']=np.where(~np.isnan(jet_new_phi), jet_new_phi, event_middphi)
    event_dphi_j2MET=(event.jet_2_phi-event.PuppiMET_phi)
    event_negative_jet2phi=ak.fill_none((event_dphi_j2MET.mask[event_dphi_j2MET<-np.pi]+2*np.pi),np.nan)
    event_negative_jet2phi=ak.fill_none((event_dphi_j2MET.mask[event_dphi_j2MET<-np.pi]+2*np.pi),np.nan)
    event_positive_jet2phi=ak.fill_none((-event_dphi_j2MET.mask[event_dphi_j2MET>np.pi]+2*np.pi),np.nan)
    event_middphi2=ak.fill_none((event_dphi_j2MET.mask[(event_dphi_j2MET<=np.pi)&(event_dphi_j2MET>=-np.pi)]),np.nan)
    jet_new_phi = np.where(~np.isnan(event_positive_jet2phi), event_positive_jet2phi, event_negative_jet2phi)
    event['dphi_jet2_PuppiMET']=np.where(~np.isnan(jet_new_phi), jet_new_phi, event_middphi2)
    event['dphi_jet2_PuppiMET']=ak.where((event['jet_2_pt']<0),(ak.ones_like(dummy)*(-1)),event['dphi_jet2_PuppiMET'])
    return event
def electron(event):
    electron=vector.obj(pt=event.electron_iso_pt,eta=event.electron_iso_eta,phi=event.electron_iso_phi,mass=event.electron_iso_mass)
    event['electron_iso_E']=electron.E
    event['electron_iso_Et']=np.sqrt(electron.mass*electron.mass+electron.pt*electron.pt)
    event['electron_iso_MET_mt']=np.sqrt(electron.mass*electron.mass+2*(event.electron_iso_Et*event.MET_pt-electron.pt*event.MET_pt))
    return event
def muon(event):
    muon=vector.obj(pt=event.muon_iso_pt,eta=event.muon_iso_eta,phi=event.muon_iso_phi,mass=event.muon_iso_mass)
    event['muon_iso_E']=muon.E
    event['muon_iso_Et']=np.sqrt(muon.mass*muon.mass+muon.pt*muon.pt)
    event['muon_iso_MET_mt']=np.sqrt(muon.mass*muon.mass+2*(event.muon_iso_Et*event.MET_pt-muon.pt*event.MET_pt))
    return event
#####

# inputfile=['/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-250_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-260_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-270_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-280_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-300_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-320_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-350_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-400_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-450_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_250to550/UL17_R_gghh_M-550_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-600_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-650_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-700_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-750_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-800_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-850_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-900_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/FH_600to1000/UL17_R_gghh_M-1000_2017/merged_nominal.parquet']
# outputfile_FH=['/eos/user/s/shsong/HHWWgg/parquet/cat7/FH250.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH260.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH270.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH280.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH300.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH320.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH350.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH400.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH450.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH550.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH600.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH650.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH700.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH750.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH800.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH850.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH900.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/FH1000.parquet']
# inputfile=['/eos/user/s/shsong/HHWWgg/parquet/bkg/fakephoton/UL17_DiPhotonJetsBox_M40_80_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/fakephoton/UL17_DiPhotonJetsBox_MGG_80toInf_2017/pp1.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/fakephoton/UL17_DiPhotonJetsBox_MGG_80toInf_2017/pp2.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/fakephoton/UL17_GJet_Pt_20to40_DoubleEMEnriched_MGG_80toInf_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/fakephoton/UL17_GJet_Pt_20toInf_DoubleEMEnriched_MGG_40to80_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/fakephoton/UL17_GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_2017/merged_nominal.parquet']
# outputfile_FH=['/eos/user/s/shsong/HHWWgg/parquet/cat7/DiPhotonJetsBox_M40_80_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiPhotonJetsBox_MGG_80toInf_20171.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/DiPhotonJetsBox_MGG_80toInf_20172.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/Gjet_Pt20_Inf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/Gjet_Pt40_Inf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/Gjet_Pt20_40_2017.parquet']
# inputfile=['/eos/user/s/shsong/HHWWgg/parquet/data_WMD/merged_nominal.parquet']
# outputfile_FH=['/eos/user/s/shsong/HHWWgg/parquet/cat7/Data_2017.parquet']
# outputfile_SL=['/eos/user/s/shsong/HHWWgg/parquet/cat2/Data_2017.parquet']
# outputfile_SL_muon=['/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/Data_2017.parquet']
# outputfile_SL_electron=['/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/Data_2017.parquet']

inputfile=['/eos/user/s/shsong/HHWWgg/parquet/SL_250to500/UL17_R_gghh_SL_M-250_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_250to500/UL17_R_gghh_SL_M-260_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_250to500/UL17_R_gghh_SL_M-270_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_250to500/UL17_R_gghh_SL_M-280_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_250to500/UL17_R_gghh_SL_M-300_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_250to500/UL17_R_gghh_SL_M-320_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_250to500/UL17_R_gghh_SL_M-350_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_250to500/UL17_R_gghh_SL_M-400_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_250to500/UL17_R_gghh_SL_M-450_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_250to500/UL17_R_gghh_SL_M-500_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_500to1000/UL17_R_gghh_SL_M-550_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_500to1000/UL17_R_gghh_SL_M-600_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_500to1000/UL17_R_gghh_SL_M-650_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_500to1000/UL17_R_gghh_SL_M-700_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_500to1000/UL17_R_gghh_SL_M-750_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_500to1000/UL17_R_gghh_SL_M-800_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_500to1000/UL17_R_gghh_SL_M-850_2017/merged_nominal.parquet', '/eos/user/s/shsong/HHWWgg/parquet/SL_500to1000/UL17_R_gghh_SL_M-900_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/SL_500to1000/UL17_R_gghh_SL_M-1000_2017/merged_nominal.parquet']
outputfile_SL=['/eos/user/s/shsong/HHWWgg/parquet/cat2/SL250.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL260.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL270.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL280.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL300.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL320.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL350.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL400.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL450.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL500.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL550.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL600.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL650.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL700.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL750.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL800.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL850.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL900.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/SL1000.parquet']
outputfile_SL_electron=['/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL250.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL260.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL270.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL280.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL300.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL320.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL350.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL400.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL450.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL500.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL550.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL600.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL650.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL700.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL750.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL800.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL850.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL900.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/SL1000.parquet']
outputfile_SL_muon=['/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL250.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL260.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL270.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL280.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL300.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL320.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL350.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL400.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL450.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL500.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL550.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL600.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL650.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL700.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL750.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL800.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL850.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL900.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/SL1000.parquet']
# inputfile=['/eos/user/s/shsong/HHWWgg/parquet/bkg/fakephoton/UL17_DiPhotonJetsBox_M40_80_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/fakephoton/UL17_DiPhotonJetsBox_MGG_80toInf_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/fakephoton/UL17_GJet_Pt_20to40_DoubleEMEnriched_MGG_80toInf_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/fakephoton/UL17_GJet_Pt_20toInf_DoubleEMEnriched_MGG_40to80_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/fakephoton/UL17_GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_2017/merged_nominal.parquet']
# outputfile_SL=['/eos/user/s/shsong/HHWWgg/parquet/cat2/DiPhotonJetsBox_M40_80_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/DiPhotonJetsBox_MGG_80toInf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/Gjet_Pt20_Inf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/Gjet_Pt40_Inf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/Gjet_Pt20_40_2017.parquet']
# outputfile_SL_muon=['/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/DiPhotonJetsBox_M40_80_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/DiPhotonJetsBox_MGG_80toInf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/Gjet_Pt20_Inf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/Gjet_Pt40_Inf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_muon/Gjet_Pt20_40_2017.parquet']
# outputfile_SL_electron=['/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/DiPhotonJetsBox_M40_80_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/DiPhotonJetsBox_MGG_80toInf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/Gjet_Pt20_Inf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/Gjet_Pt40_Inf_2017.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2_electron/Gjet_Pt20_40_2017.parquet']


# inputfile=['/eos/user/s/shsong/HHWWgg/parquet/bkg/WJets/UL17_WWG_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/WJets/UL17_W1JetsToLNu_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/WJets/UL17_W2JetsToLNu_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/WJets/UL17_W3JetsToLNu_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/WJets/UL17_W4JetsToLNu_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/WJets/UL17_WWTo1L1Nu2Q_4f_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/WJets/WJetsToQQ_HT-200to400_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/WJets/WJetsToQQ_HT-600to800_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/WJets/WJetsToQQ_HT-400to600_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/WJets/WJetsToQQ_HT-800toInf_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/TTbar/UL17_TTGG_0Jets/UL17_TTGG_0Jets_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/TTbar/UL17_TTGJets/UL17_TTGJets_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/TTbar/UL17_TTJets/UL17_TTJets_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/TTbar/UL17_ttWJets/UL17_ttWJets_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/QCD/UL17_QCD_Pt-30to40_MGG-80toInf_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/QCD/UL17_QCD_Pt-30toInf_MGG-40to80_2017/merged_nominal.parquet','/eos/user/s/shsong/HHWWgg/parquet/bkg/QCD/UL17_QCD_Pt-40ToInf_MGG-80ToInf_2017/merged_nominal.parquet']
# outputfile_FH=['/eos/user/s/shsong/HHWWgg/parquet/cat7/WWG.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/W1JetsToLNu.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/W2JetsToLNu.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/W3JetsToLNu.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/W4JetsToLNu.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/WWTo1L1Nu2Q.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/WJetsToQQ200to400.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/WJetsToQQ600to800.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/WJetsToQQ400to600.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/WJetsToQQ800toInf.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/TTGG_0Jets.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/TTGJets.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/TTJets.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/ttWJets.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/QCDpt30To40.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/QCDpt30ToInf.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat7/QCDpt40ToInf.parquet']

# outputfile_SL=['/eos/user/s/shsong/HHWWgg/parquet/cat2/WWG.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/W1JetsToLNu.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/W2JetsToLNu.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/W3JetsToLNu.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/W4JetsToLNu.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/WWTo1L1Nu2Q.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/WJetsToQQ200to400.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/WJetsToQQ600to800.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/WJetsToQQ400to600.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/WJetsToQQ800toInf.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/TTGG_0Jets.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/TTGJets.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/TTJets.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/ttWJets.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/QCDpt30To40.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/QCDpt30ToInf.parquet','/eos/user/s/shsong/HHWWgg/parquet/cat2/QCDpt40ToInf.parquet']
i=0
for file in inputfile:
    eventSL, eventFH=load_parquet(file)
    print("Got event")
    # eventFH=calclulate_W_info(eventFH)
    # print('Got W info')
    # eventFH=scaled_diphoton_info(eventFH)
    # print('Got scaled_diphoton info')
    # eventFH=E_resolution(eventFH)
    # print('Got E resolution')
    # eventFH=calculate_bscore(eventFH)
    # print('Got b score')
    # eventFH=getCosThetaStar(eventFH)
    # print('Got getCosThetaStar')
    # eventFH=calculate_dR_gg_4jets(eventFH)
    # print('Got dr gg 4jets')
    # eventFH=calculate_dR_4jets(eventFH)
    # print('Got dr 4jets')

    # eventFH=costheta1(eventFH)
    # print('Got CosTheta1')
    # eventFH=costheta2(eventFH)
    # print('Got CosTheta2')
    # eventFH=get_minmaxID(eventFH)
    # print('Got get_minmaxID')
    # # eventFH=calculate_sphericity(eventFH)
    # # print('Got sphericity')
    # ak.to_parquet(eventFH, outputfile_FH[i])
    # parquet_to_root(outputfile_FH[i],outputfile_FH[i].replace("parquet","root"),treename="cat7",verbose=False)


    eventSL=calculate_Wjj(eventSL)
    print('Got W info')
    eventSL=scaled_diphoton_info(eventSL)
    print('Got photon info')
    eventSL=E_resolution(eventSL)
    print('Got E resolution')
    eventSL=lepton(eventSL)
    # eventSL=get_minmaxID(eventSL)
    print('Got get_minmaxID')
    SL_muon=eventSL[eventSL.nGoodisomuons==1]
    SL_muon=muon(SL_muon)
    SL_electron=eventSL[eventSL.nGoodisoelectrons==1]
    SL_electron=electron(SL_electron)
    print('Got electron info')
    ak.to_parquet(eventSL, outputfile_SL[i])
    ak.to_parquet(SL_muon, outputfile_SL_muon[i])
    ak.to_parquet(SL_electron, outputfile_SL_electron[i])
    parquet_to_root(outputfile_SL[i],outputfile_SL[i].replace("parquet","root"),treename="cat2",verbose=False)
    parquet_to_root(outputfile_SL_muon[i],outputfile_SL_muon[i].replace("parquet","root"),treename="cat2",verbose=False)
    parquet_to_root(outputfile_SL_electron[i],outputfile_SL_electron[i].replace("parquet","root"),treename="cat2",verbose=False)
    i=i+1



