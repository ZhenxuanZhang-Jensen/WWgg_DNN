import awkward as ak
import numpy as np
import vector
vector.register_awkward()
import pandas as pd
from parquet_to_root import parquet_to_root

def load_parquet(fname):
    events=ak.from_parquet(fname)
    return events
def calclulate_W_info(event):
    leadjet=vector.obj(pt=event.jet_1_pt,eta=event.jet_1_eta,phi=event.jet_1_phi,mass=event.jet_1_mass)
    event['jet_1_E']=leadjet.E
    subleadjet=vector.obj(pt=event.jet_2_pt,eta=event.jet_2_eta,phi=event.jet_2_phi,mass=event.jet_2_mass)
    event['jet_2_E']=subleadjet.E
    W1=leadjet+subleadjet
    event['W1_pt']=W1.pt
    event['W1_eta']=W1.eta
    event['W1_phi']=W1.phi
    event['W1_mass']=W1.mass
    event['W1_E']=W1.E

    subsubleadjet=vector.obj(pt=event.jet_3_pt,eta=event.jet_3_eta,phi=event.jet_3_phi,mass=event.jet_3_mass)
    event['jet_3_E']=subsubleadjet.E
    subsubsubleadjet=vector.obj(pt=event.jet_4_pt,eta=event.jet_4_eta,phi=event.jet_4_phi,mass=event.jet_4_mass)
    event['jet_4_E']=subsubsubleadjet.E
    W2=subsubleadjet+subsubsubleadjet
    event['W2_pt']=W2.pt
    event['W2_eta']=W2.eta
    event['W2_phi']=W2.phi
    event['W2_mass']=W2.mass
    event['W2_E']=W2.E

    WW=W1+W2
    event['WW_pt']=WW.pt
    event['WW_eta']=WW.eta
    event['WW_phi']=WW.phi
    event['WW_mass']=WW.mass
    event['WW_E']=WW.E

    return event

def scaled_diphoton_info(event):
    event['scaled_leadphoton_pt']=event['LeadPhoton_pt']/event['Diphoton_mass']
    event['scaled_subleadphoton_pt']=event['SubleadPhoton_pt']/event['Diphoton_mass']
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

def mass_resolution(event):
    leadpho_px=event['LeadPhoton_pt']*np.cos(event['LeadPhoton_phi'])
    leadpho_py=event['LeadPhoton_pt']*np.sin(event['LeadPhoton_phi'])
    leadpho_pz=event['LeadPhoton_pt']*np.sinh(event['LeadPhoton_eta'])
    event['LeadPhoton_E']=np.sqrt(np.array(leadpho_px)*np.array(leadpho_px)+np.array(leadpho_py)*np.array(leadpho_py)+np.array(leadpho_pz)*np.array(leadpho_pz)-event['LeadPhoton_mass']*event['LeadPhoton_mass'])
    event['LeadPhoton_sigEoverE']=event['LeadPhoton_energyErr']/event['LeadPhoton_E']

    subleadpho_px=event['SubleadPhoton_pt']*np.cos(event['SubleadPhoton_phi'])
    subleadpho_py=event['SubleadPhoton_pt']*np.sin(event['SubleadPhoton_phi'])
    subleadpho_pz=event['SubleadPhoton_pt']*np.sinh(event['SubleadPhoton_eta'])
    event['SubleadPhoton_E']=np.sqrt(np.array(subleadpho_px)*np.array(subleadpho_px)+np.array(subleadpho_py)*np.array(subleadpho_py)+np.array(subleadpho_pz)*np.array(subleadpho_pz)-event['SubleadPhoton_mass']*event['SubleadPhoton_mass'])
    event['SubleadPhoton_sigEoverE']=event['SubleadPhoton_energyErr']/event['SubleadPhoton_E']
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
    vx=HH.px / (HH.mass * HH.beta)
    vy=HH.py / (HH.mass * HH.beta)
    vz=HH.pz / (HH.mass * HH.beta)
    event['costhetastar']=vz
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
    event['mindR_gg_4jets']=diphoton_jetdR[ak.argsort(diphoton_jetdR)][:,0]
    event['maxdR_gg_4jets']=diphoton_jetdR[ak.argsort(diphoton_jetdR)][:,7]
    return event

def calculate_dR_4jets(event):
    jet1=ak.zip({"pt":event.jet_1_pt,"eta":event.jet_1_eta,"phi":event.jet_1_phi,"phi":event.jet_1_phi},with_name="Momentum4D")
    jet2=ak.zip({"pt":event.jet_2_pt,"eta":event.jet_2_eta,"phi":event.jet_2_phi,"phi":event.jet_2_phi},with_name="Momentum4D")
    jet3=ak.zip({"pt":event.jet_3_pt,"eta":event.jet_3_eta,"phi":event.jet_3_phi,"phi":event.jet_3_phi},with_name="Momentum4D")
    jet4=ak.zip({"pt":event.jet_4_pt,"eta":event.jet_4_eta,"phi":event.jet_4_phi,"phi":event.jet_4_phi},with_name="Momentum4D")

    j1j2dr=ak.unflatten(jet1.deltaR(jet2), counts=1)
    j1j3dr=ak.unflatten(jet1.deltaR(jet3), counts=1)
    j1j4dr=ak.unflatten(jet1.deltaR(jet4), counts=1)
    j2j3dr=ak.unflatten(jet2.deltaR(jet3), counts=1)
    j2j4dr=ak.unflatten(jet2.deltaR(jet4), counts=1)
    j3j4dr=ak.unflatten(jet3.deltaR(jet4), counts=1)

    jjdR=ak.concatenate([j1j2dr,j1j3dr,j1j4dr,j2j3dr,j2j4dr,j3j4dr],axis=1)
    event['maxdR_4jets']=jjdR[ak.argsort(jjdR)][:,5]
    event['mindR_4jets']=jjdR[ak.argsort(jjdR)][:,0]
    return event
def add_sale_factor(event,sclae_factor):
    event['weight_central']=sclae_factor*event.weight_central
    return event

inputfile=["/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m250.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m260.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m270.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m280.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m300.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m320.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m350.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m400.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m450.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m550.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m600.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m650.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m700.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m750.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m800.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m850.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m900.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m1000.parquet"]
# inputfile=["/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DiphotonJetbox.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DatadrivenQCD.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/Data.parquet","/eos/user/s/shsong/combined_WWgg/DNN/cat2/m600.parquet"]
# scalefactor=[1.43242,1.15197,1,1]
# outputfile=["/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DiphotonJetbox_reweight.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DatadrivenQCD_reweight.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/Data_new.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/m600.parquet"]
outputfile=["/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m250.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m260.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m270.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m280.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m300.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m320.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m350.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m400.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m450.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m550.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m600.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m650.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m700.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m750.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m800.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m850.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m900.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m1000.parquet"]

i=0
for file in inputfile:
    event=load_parquet(file)
    event=calclulate_W_info(event)
    event=scaled_diphoton_info(event)
    event=mass_resolution(event)
    event=calculate_bscore(event)
    event=getCosThetaStar(event)
    event=calculate_dR_gg_4jets(event)
    event=calculate_dR_4jets(event)
    # event=add_sale_factor(event,scalefactor[i])
    ak.to_parquet(event, outputfile[i])
    i=i+1

# parquet_to_root("/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DiphotonJetbox_reweight.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DiphotonJetbox_reweight.root",treename="cat2",verbose=False)
# parquet_to_root("/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DatadrivenQCD_reweight.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DatadrivenQCD_reweight.root",treename="cat2",verbose=False)
# parquet_to_root("/eos/user/s/shsong/combined_WWgg/datadriven/cat2/Data_new.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/Data_new.root",treename="cat2",verbose=False)
# parquet_to_root("/eos/user/s/shsong/combined_WWgg/datadriven/cat2/m600.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/m600.root",treename="cat2",verbose=False)