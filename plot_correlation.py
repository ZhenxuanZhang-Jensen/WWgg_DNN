import pandas as pd

import numpy as np

import seaborn as sns
data_set=pd.read_csv('/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat2_40/dnn.csv',index_col=0)
training_columns=["new_weight","scaled_leadphoton_pt","scaled_subleadphoton_pt","WW_pt","W1_mass","WW_mass","mindR_gg_4jets","maxdR_gg_4jets","mindR_4jets","maxdR_4jets","W2_mass","jet_1_pt","jet_2_pt","jet_4_pt","jet_4_E","jet_3_E","jet_2_E","jet_1_E","jet_3_pt","sum_two_max_bscore","LeadPhoton_eta","SubleadPhoton_eta","W1_pt","W2_pt","jet_1_eta","jet_2_eta","jet_3_eta","jet_4_eta","costhetastar","LeadPhoton_sigEoverE","SubleadPhoton_sigEoverE","Diphoton_pt","Diphoton_minID","Diphoton_maxID","Signal_Mass","nGoodAK4jets","sphericity","costheta1","costheta2"]
allFeatures = ["Diphoton_dR"]

df = pd.DataFrame({
"scaled_leadphoton_pt":data_set['scaled_leadphoton_pt'],
"scaled_subleadphoton_pt":data_set['scaled_subleadphoton_pt'],
"WW_pt":data_set['WW_pt'],
"W1_mass":data_set['W1_mass'],
"WW_mass":data_set['WW_mass'],
"mindR_gg_4jets":data_set['mindR_gg_4jets'],
"maxdR_gg_4jets":data_set['maxdR_gg_4jets'],
"mindR_4jets":data_set['mindR_4jets'],
"maxdR_4jets":data_set['maxdR_4jets'],
"W2_mass":data_set['W2_mass'],
"jet_1_pt":data_set['jet_1_pt'],
"jet_2_pt":data_set['jet_2_pt'],
"jet_4_pt":data_set['jet_4_pt'],
"jet_4_E":data_set['jet_4_E'],
"jet_3_E":data_set['jet_3_E'],
"jet_2_E":data_set['jet_2_E'],
"jet_1_E":data_set['jet_1_E'],
"jet_3_pt":data_set['jet_3_pt'],
"sum_two_max_bscore":data_set['sum_two_max_bscore'],
"LeadPhoton_eta":data_set['LeadPhoton_eta'],
"SubleadPhoton_eta":data_set['SubleadPhoton_eta'],
"W1_pt":data_set['W1_pt'],
"W2_pt":data_set['W2_pt'],
"jet_1_eta":data_set['jet_1_eta'],
"jet_2_eta":data_set['jet_2_eta'],
"jet_3_eta":data_set['jet_3_eta'],
"jet_4_eta":data_set['jet_4_eta'],
"costhetastar":data_set['costhetastar'],
"LeadPhoton_sigEoverE":data_set['LeadPhoton_sigEoverE'],
"SubleadPhoton_sigEoverE":data_set['SubleadPhoton_sigEoverE'],
"Diphoton_pt":data_set['Diphoton_pt'],
"Diphoton_minID":data_set['Diphoton_minID'],
"Diphoton_maxID":data_set['Diphoton_maxID'],
"Signal_Mass":data_set['Signal_Mass'],
"nGoodAK4jets":data_set['nGoodAK4jets'],
"sphericity":data_set['sphericity'],
"costheta1":data_set['costheta1'],
"costheta2":data_set['costheta2'],
"Diphoton_dR":data_set['Diphoton_dR']})

corr = df.corr()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(30,25))

sns.heatmap(corr, cmap='Blues', annot=True)
fig.savefig("/hpcfs/cms/cmsgpu/shaoweisong/DNN/corr2D.png")
