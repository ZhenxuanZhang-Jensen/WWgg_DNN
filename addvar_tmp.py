import awkward as ak
import numpy as np
import vector
vector.register_awkward()
import pandas as pd
import sys
from parquet_to_root import parquet_to_root

def load_parquet(fname):
    events=ak.from_parquet(fname)
    return events
def split_electron_category(event):
    event=event[event.category==3]
    event=event[event.electron_pt>0]
    return event
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
    event['dphi_j1j2']=abs(event.jet_1_phi-event.jet_2_phi)
    event['dphi_j1j3']=abs(event.jet_1_phi-event.jet_3_phi)
    event['dphi_j1j4']=abs(event.jet_1_phi-event.jet_4_phi)
    event['dphi_j2j3']=abs(event.jet_2_phi-event.jet_3_phi)
    event['dphi_j2j4']=abs(event.jet_2_phi-event.jet_4_phi)
    event['dphi_j3j4']=abs(event.jet_3_phi-event.jet_4_phi)

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

def E_resolution(event):
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
    # vx=HH.px / (HH.mass * HH.beta)
    # vy=HH.py / (HH.mass * HH.beta)
    # vz=HH.pz / (HH.mass * HH.beta)
    # event['costhetastar']=vz
    p= np.sqrt(HH.px**2 + HH.py**2 + HH.pz**2)
    event['costhetastar']=HH.pz/p

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
def calculate_photon_info(event):
    event['scaled_leadphoton_pt']=event['LeadPhoton_pt']/event['Diphoton_mass']
    event['scaled_subleadphoton_pt']=event['SubleadPhoton_pt']/event['Diphoton_mass']
    leadphoton=vector.obj(pt=event.LeadPhoton_pt,eta=event.LeadPhoton_eta,phi=event.LeadPhoton_phi,mass=event.LeadPhoton_mass)
    subleadphoton=vector.obj(pt=event.SubleadPhoton_pt,eta=event.SubleadPhoton_eta,phi=event.SubleadPhoton_phi,mass=event.SubleadPhoton_mass)
    event['LeadPhoton_E_over_mass']=leadphoton.E/event.Diphoton_mass
    event['SubleadPhoton_E_over_mass']=subleadphoton.E/event.Diphoton_mass
    return event
def calculate_Wjj(event):
    leadjet=vector.obj(pt=event.jet_1_pt,eta=event.jet_1_eta,phi=event.jet_1_phi,mass=event.jet_1_mass)
    event['jet_1_E']=leadjet.E
    subleadjet=vector.obj(pt=event.jet_2_pt,eta=event.jet_2_eta,phi=event.jet_2_phi,mass=event.jet_2_mass)
    event['jet_2_E']=subleadjet.E
    event['dphi_jj']=abs(event.jet_1_phi-event.jet_2_phi)
    W1=leadjet+subleadjet
    event['W1_pt']=W1.pt
    event['W1_eta']=W1.eta
    event['W1_phi']=W1.phi
    event['W1_mass']=W1.mass
    event['W1_E']=W1.E
    return event
def lepton(event):
    lepton_pt=ak.concatenate([ak.unflatten(event.electron_pt,counts=1),ak.unflatten(event.muon_pt,counts=1)],axis=1)
    lepton_eta=ak.concatenate([ak.unflatten(event.electron_eta,counts=1),ak.unflatten(event.muon_eta,counts=1)],axis=1)
    lepton_phi=ak.concatenate([ak.unflatten(event.electron_phi,counts=1),ak.unflatten(event.muon_phi,counts=1)],axis=1)
    lepton_mass=ak.concatenate([ak.unflatten(event.electron_mass,counts=1),ak.unflatten(event.muon_mass,counts=1)],axis=1)
    lepton=vector.obj(pt=ak.flatten(lepton_pt[lepton_pt!=-999],axis=1),eta=ak.flatten(lepton_eta[lepton_eta!=-999],axis=1),phi=ak.flatten(lepton_phi[lepton_phi!=-999],axis=1),mass=ak.flatten(lepton_mass[lepton_mass!=-999],axis=1))
    event['lepton_pt']=lepton.pt
    event['lepton_eta']=lepton.eta
    event['lepton_phi']=lepton.phi
    event['lepton_pt']=lepton.mass
    event['lepton_E']=lepton.E
    lepton_Et=np.sqrt(lepton.mass*lepton.mass+lepton.pt*lepton.pt)
    # lepton_Et=(lepton.E/np.sqrt(lepton.E*lepton.E-lepton.mass*lepton.mass))*lepton.pt
    Mt=np.sqrt(lepton.mass*lepton.mass+2*(lepton_Et*event.MET_pt-lepton.pt*event.MET_pt))
    event['Mt_lepMET']=Mt

    dphi_jet1_MET=abs(event.MET_phi-event.jet_1_phi)   
    dphi_jet2_MET=abs(event.MET_phi-event.jet_2_phi)  
    event['mindphi_MET_jet'] = np.minimum(dphi_jet1_MET,dphi_jet2_MET)
    event['maxdphi_MET_jet'] = np.maximum(dphi_jet1_MET,dphi_jet2_MET)
    return event
def electron(event):
    electron=vector.obj(pt=event.electron_pt,eta=event.electron_eta,phi=event.electron_phi,mass=event.electron_mass)
    event['electron_E']=electron.E
    event['electron_Et']=np.sqrt(electron.mass*electron.mass+electron.pt*electron.pt)
    event['electron_MET_mt']=np.sqrt(electron.mass*electron.mass+2*(event.electron_Et*event.MET_pt-electron.pt*event.MET_pt))
    return event


#initial signal parquet
# inputfile=["/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m250.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m260.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m270.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m280.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m300.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m320.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m350.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m400.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m450.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m550.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m600.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m650.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m700.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m750.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m800.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m850.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m900.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/category2/DNN/m1000.parquet"]
# inputfile=["/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DiphotonJetbox_reweight.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DatadrivenQCD_reweight.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/Data_new.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/m600.parquet"]
type = sys.argv[1]

# scalefactor=[1.43242,1.15197,1,1]
# outputfile=["/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DiphotonJetbox_reweight.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DatadrivenQCD_reweight.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/Data_new.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/m600.parquet"]
# inputfile=["/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m250.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m260.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m270.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m280.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m300.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m320.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m350.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m400.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m450.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m550.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m600.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m650.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m700.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m750.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m800.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m850.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m900.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m1000.parquet"]
# outputfile=["/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m250.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m260.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m270.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m280.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m300.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m320.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m350.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m400.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m450.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m550.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m600.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m650.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m700.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m750.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m800.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m850.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m900.parquet","/eos/user/s/shsong/combined_WWgg/datadriven/cat2/DNN/m1000.parquet"]
# inputfile=["/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-250_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-260_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-270_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-280_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-300_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-320_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-350_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-400_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-450_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-550_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-600_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-650_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-700_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-750_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-800_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-850_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-900_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-1000_2017/merged_nominal.parquet","/eos/user/s/shsong/combined_WWgg/parquet/datadriven/category3/dd/Data_e.parquet","/eos/user/s/shsong/combined_WWgg/parquet/datadriven/category3/dd/DatadrivenQCD_reweight.parquet","/eos/user/s/shsong/combined_WWgg/parquet/datadriven/category3/dd/cat3_DiPhotonJetsBox_e.parquet"]
inputfile=['/eos/user/s/shsong/HH_WWgg/parquet/bkg/DiphotonJets/UL17_DiPhotonJetsBox_M40_80_2017/merged_nominal.parquet']
outputfile=['/eos/user/s/shsong/HH_WWgg/root/bkg/DiPhotonJetsBox_M40_80_2017_cat7.parquet']
# outputfile=["/eos/user/s/shsong/combined_WWgg/DNN/category3/m250.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m260.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m270.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m280.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m300.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m320.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m350.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m400.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m450.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m550.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m600.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m650.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m700.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m750.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m800.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m850.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m900.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/m1000.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/Data_e.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/QCD_e.parquet","/eos/user/s/shsong/combined_WWgg/DNN/category3/DiphotonJetsBox_e.parquet"]
# inputfile=["/eos/user/s/shsong/combined_WWgg/parquet/sig_latest/WWgg/UL17_R_gghh_SL_M-250_2017/merged_nominal.parquet"]
# outputfile=["/eos/user/s/shsong/combined_WWgg/DNN/category3/m250.parquet"]
i=0
if type=="FH":
    for file in inputfile:
        event=load_parquet(file)
        event=calclulate_W_info(event)
        event=scaled_diphoton_info(event)
        event=E_resolution(event)
        event=calculate_bscore(event)
        event=getCosThetaStar(event)
        event=calculate_dR_gg_4jets(event)
        event=calculate_dR_4jets(event)
        event=getCosThetaStar(event)
        event=costheta1(event)
        event=costheta2(event)
        event=calculate_sphericity(event)
        # event=add_sale_factor(event,scalefactor[i])
        ak.to_parquet(event, outputfile[i])
        i=i+1
if type=="SL":
    for file in inputfile:
        event=load_parquet(file)
        event=split_electron_category(event)
        event=calculate_Wjj(event)
        event=calculate_photon_info(event)
        event=E_resolution(event)
        event=electron(event)
        event=select_jet(event)
        ak.to_parquet(event, outputfile[i])
        parquet_to_root(outputfile[i],outputfile[i].replace("parquet","root"),treename="cat7",verbose=False)
        i=i+1



