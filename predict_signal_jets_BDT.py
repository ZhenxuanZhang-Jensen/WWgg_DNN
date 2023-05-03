import awkward as ak
import pandas as pd
# application part
# load parquet
events = read_files("/eos/user/z/zhenxuan/hhwwgg_parquet/FH_channel/hhwwgg_signal_FH_custom_test/UL17_R_gghh_M-400_2017/merged_nominal.parquet")
# get the 4D info
jet = ak.zip({
    'pt': events.SelectedJet_pt,
    'eta': events.SelectedJet_eta,
    'phi': events.SelectedJet_phi,
    'mass': events.SelectedJet_mass,
    'btag': events.SelectedJet_btagDeepFlavB
}, with_name="Momentum4D")
Gen_HWW_4q = ak.zip({
    'pt': events.GENHWW_qqqq_pt,
    'eta': events.GENHWW_qqqq_eta,
    'phi': events.GENHWW_qqqq_phi,
    'mass': events.GENHWW_qqqq_mass,
}, with_name="Momentum4D")
Photon1 = ak.zip({
    'pt': events.LeadPhoton_pt,
    'eta': events.LeadPhoton_eta,
    'phi': events.LeadPhoton_phi,
    'mass': events.LeadPhoton_mass,
}, with_name="Momentum4D")
Photon2 = ak.zip({
    'pt': events.SubleadPhoton_pt,
    'eta': events.SubleadPhoton_eta,
    'phi': events.SubleadPhoton_phi,
    'mass': events.SubleadPhoton_mass,
}, with_name="Momentum4D")
#load model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
pair = ak.combinations(jet, axis=-1,  n=4)
df_sig = pd.DataFrame()
for i in range(10):
# for i in range(len(pair)):
    tmp_score = 0
    for j in range(len(pair[i])):
        df = pd.DataFrame()
        df.at[j,'jet_1_pt'] = pair[i][j].slot0.pt
        df.at[j,'jet_2_pt'] = pair[i][j].slot1.pt
        df.at[j,'jet_3_pt'] = pair[i][j].slot2.pt
        df.at[j,'jet_4_pt'] = pair[i][j].slot3.pt

        df.at[j,'jet_1_eta'] = pair[i][j].slot0.eta
        df.at[j,'jet_2_eta'] = pair[i][j].slot1.eta
        df.at[j,'jet_3_eta'] = pair[i][j].slot2.eta
        df.at[j,'jet_4_eta'] = pair[i][j].slot3.eta
        
        df.at[j,'jet_1_phi'] = pair[i][j].slot0.phi
        df.at[j,'jet_2_phi'] = pair[i][j].slot1.phi
        df.at[j,'jet_3_phi'] = pair[i][j].slot2.phi
        df.at[j,'jet_4_phi'] = pair[i][j].slot3.phi
        
        df.at[j,'jet_1_E'] = pair[i][j].slot0.E
        df.at[j,'jet_2_E'] = pair[i][j].slot1.E
        df.at[j,'jet_3_E'] = pair[i][j].slot2.E
        df.at[j,'jet_4_E'] = pair[i][j].slot3.E
        
        df.at[count,'jet_1_btagDeepFlavB'] = pair[i][j].slot0.btag
        df.at[count,'jet_2_btagDeepFlavB'] = pair[i][j].slot1.btag
        df.at[count,'jet_3_btagDeepFlavB'] = pair[i][j].slot2.btag
        df.at[count,'jet_4_btagDeepFlavB'] = pair[i][j].slot3.btag
        
        eigvals, eigvecs = momentum_tensor([pair[i][j].slot0,pair[i][j].slot1,pair[i][j].slot2,pair[i][j].slot3])
        spher_ = sphericity(eigvals)
        df.at[j,'sphericity'] = spher_
        
        df.at[j,'Diphoton_pt'] = events['Diphoton_pt'][i]
        df.at[j,'Diphoton_dR'] = events['Diphoton_dR'][i]
        
        df.at[j,'dRj1_photon1'] = Photon1[i].deltaR(pair[i][j].slot0)
        df.at[j,'dRj2_photon1'] = Photon1[i].deltaR(pair[i][j].slot1)
        df.at[j,'dRj3_photon1'] = Photon1[i].deltaR(pair[i][j].slot2)
        df.at[j,'dRj4_photon1'] = Photon1[i].deltaR(pair[i][j].slot3)
        
        df.at[j,'dRj1_photon2'] = Photon2[i].deltaR(pair[i][j].slot0)
        df.at[j,'dRj2_photon2'] = Photon2[i].deltaR(pair[i][j].slot1)
        df.at[j,'dRj3_photon2'] = Photon2[i].deltaR(pair[i][j].slot2)
        df.at[j,'dRj4_photon2'] = Photon2[i].deltaR(pair[i][j].slot3)
        score = model.predict_proba(df[input_features].values)[0][1]
        # get the higgest score one
        if (score > tmp_score):
            tmp_socre = score
            index_of_signal = j
    # save 4 signal jets variables in a dataframe then save it in parquet
    df_sig.at[i,"signal_jet_1_pt"] = pair[i][index_of_signal].slot0.pt
    df_sig.at[i,"signal_jet_2_pt"] = pair[i][index_of_signal].slot1.pt
    df_sig.at[i,"signal_jet_3_pt"] = pair[i][index_of_signal].slot2.pt
    df_sig.at[i,"signal_jet_4_pt"] = pair[i][index_of_signal].slot3.pt

    df_sig.at[i,"signal_jet_1_eta"] = pair[i][index_of_signal].slot0.eta
    df_sig.at[i,"signal_jet_2_eta"] = pair[i][index_of_signal].slot1.eta
    df_sig.at[i,"signal_jet_3_eta"] = pair[i][index_of_signal].slot2.eta
    df_sig.at[i,"signal_jet_4_eta"] = pair[i][index_of_signal].slot3.eta
    
    df_sig.at[i,"signal_jet_1_phi"] = pair[i][index_of_signal].slot0.phi
    df_sig.at[i,"signal_jet_2_phi"] = pair[i][index_of_signal].slot1.phi
    df_sig.at[i,"signal_jet_3_phi"] = pair[i][index_of_signal].slot2.phi
    df_sig.at[i,"signal_jet_4_phi"] = pair[i][index_of_signal].slot3.phi

    df_sig.at[i,"signal_jet_1_E"] = pair[i][index_of_signal].slot0.E
    df_sig.at[i,"signal_jet_2_E"] = pair[i][index_of_signal].slot1.E
    df_sig.at[i,"signal_jet_3_E"] = pair[i][index_of_signal].slot2.E
    df_sig.at[i,"signal_jet_4_E"] = pair[i][index_of_signal].slot3.E

    df_sig.at[i,"signal_jet_1_btag"] = pair[i][index_of_signal].slot0.btag
    df_sig.at[i,"signal_jet_2_btag"] = pair[i][index_of_signal].slot1.btag
    df_sig.at[i,"signal_jet_3_btag"] = pair[i][index_of_signal].slot2.btag
    df_sig.at[i,"signal_jet_4_btag"] = pair[i][index_of_signal].slot3.btag
# save the signal 4 jets variables in the parquet
for i in range(len(df_sig.columns)):
    events[df_sig.columns[i]] = df_sig[df_sig.columns[i]]
