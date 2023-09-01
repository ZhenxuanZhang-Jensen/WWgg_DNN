import pandas as pd

import numpy as np

import seaborn as sns
# data_set=pd.read_csv('/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat2_40/dnn.csv',index_col=0)
data_set=pd.read_csv('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/dnn.csv',index_col=0)
# training_columns=["new_weight","scaled_leadphoton_pt","scaled_subleadphoton_pt","WW_pt","W1_mass","WW_mass","maxdR_gg_4jets","mindR_4jets","W2_mass","jet_1_pt","jet_2_pt","jet_4_pt","jet_4_E","jet_3_E","jet_2_E","jet_1_E","jet_3_pt","sum_two_max_bscore","LeadPhoton_eta","SubleadPhoton_eta","W1_pt","W2_pt","jet_1_eta","jet_2_eta","jet_3_eta","jet_4_eta","costhetastar","LeadPhoton_sigEoverE","SubleadPhoton_sigEoverE","Diphoton_pt","Diphoton_minID","Diphoton_maxID","nGoodAK4jets","costheta1","costheta2","Signal_Mass","Diphoton_mass"]

allFeatures = ["Diphoton_dR"]

df = pd.DataFrame({"Diphoton_mass":data_set["Diphoton_mass"],
"Diphoton_pt":data_set["Diphoton_pt"],
"Diphoton_eta":data_set["Diphoton_eta"],
"Diphoton_phi":data_set["Diphoton_phi"],
"scaled_leadphoton_pt":data_set["scaled_leadphoton_pt"],
"scaled_subleadphoton_pt":data_set["scaled_subleadphoton_pt"],
"Diphoton_minID":data_set["Diphoton_minID"],
"Diphoton_maxID":data_set["Diphoton_maxID"],
"LeadPhoton_eta":data_set["LeadPhoton_eta"],
"SubleadPhoton_eta":data_set["SubleadPhoton_eta"],
"LeadPhoton_sigEoverE":data_set["LeadPhoton_sigEoverE"],
"SubleadPhoton_sigEoverE":data_set["SubleadPhoton_sigEoverE"],
"Diphoton_mass_resolution":data_set["Diphoton_mass_resolution"],
"WW_E":data_set["WW_E"],
"WW_pt":data_set["WW_pt"],
"WW_mass":data_set["WW_mass"],
"WW_eta":data_set["WW_eta"],
"WW_phi":data_set["WW_phi"],
"W1_mass":data_set["W1_mass"],
"W1_pt":data_set["W1_pt"],
"W1_eta":data_set["W1_eta"],
"W1_phi":data_set["W1_phi"],
"W1_E":data_set["W1_E"],
"W2_mass":data_set["W2_mass"],
"W2_pt":data_set["W2_pt"],
"maxdR_gg_4jets":data_set["maxdR_gg_4jets"],
"mindR_gg_4jets":data_set["mindR_gg_4jets"],
"maxdR_4jets":data_set["maxdR_4jets"],
"mindR_4jets":data_set["mindR_4jets"],
"jet_1_pt":data_set["jet_1_pt"],
"jet_1_E":data_set["jet_1_E"],
"jet_1_eta":data_set["jet_1_eta"],
"jet_1_phi":data_set["jet_1_phi"],
"jet_1_btagDeepFlavB":data_set["jet_1_btagDeepFlavB"],
"jet_2_pt":data_set["jet_2_pt"],
"jet_2_E":data_set["jet_2_E"],
"jet_2_eta":data_set["jet_2_eta"],
"jet_2_phi":data_set["jet_2_phi"],
"jet_2_btagDeepFlavB":data_set["jet_2_btagDeepFlavB"],
"jet_3_pt":data_set["jet_3_pt"],
"jet_3_E":data_set["jet_3_E"],
"jet_3_eta":data_set["jet_3_eta"],
"jet_3_phi":data_set["jet_3_phi"],
"jet_3_btagDeepFlavB":data_set["jet_3_btagDeepFlavB"],
"jet_4_pt":data_set["jet_4_pt"],
"jet_4_E":data_set["jet_4_E"],
"jet_4_eta":data_set["jet_4_eta"],
"jet_4_phi":data_set["jet_4_phi"],
"jet_4_btagDeepFlavB":data_set["jet_4_btagDeepFlavB"],
"sum_two_max_bscore":data_set["sum_two_max_bscore"],
"costhetastar":data_set["costhetastar"],
"nGoodAK4jets":data_set["nGoodAK4jets"],
"costheta1":data_set["costheta1"],
"costheta2":data_set["costheta2"],
"Signal_Mass":data_set["Signal_Mass"]
})

corr = df.corr()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(50,45))

sns.heatmap(corr, cmap='Blues', annot=True)
fig.savefig("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/corr2D.png")
