import ROOT
from array import array
import awkward 
import numpy
from parquet_to_root import parquet_to_root
DataFile=awkward.from_parquet("/eos/user/s/shsong/HiggsDNA/data2017/merged_nominal.parquet")
DataFile=DataFile[(DataFile.category==2)&(DataFile.nGoodisoelectrons==1)]
# DataFile=DataFile[(DataFile.category==2)&(DataFile.nGoodisomuons==1)]
sideband_min=numpy.min(DataFile[DataFile.is_passPhotonMVA90].Diphoton_minID)
print(sideband_min)
GJets1=ROOT.TFile.Open("/eos/user/s/shsong/HiggsDNA/root/bkg/GJet.root")
# GJets1=ROOT.TFile.Open("/eos/user/s/shsong/HiggsDNA/root/cat2/GJet_m.root")

GJets1_tree=GJets1.Get("Gjet")
cuts="(Diphoton_mass<115 || Diphoton_mass>135)*(minID_genPartFlav!=1)*(category==4)"
h_minphotonID=ROOT.TH1F("h_minphotonID_gjet","h_minphotonID_gjet",19,-0.9,1)
GJets1_tree.Project("h_minphotonID_gjet","Diphoton_minID","weight_central*"+cuts)
photonIDPDF_fake=ROOT.TF1("photonIDPDF_fake","pol7",-0.9,1.)
h_minphotonID.Fit(photonIDPDF_fake,"R")
c1=ROOT.TCanvas("c1","c1",600,800)
h_minphotonID.Draw("E1")
c1.SaveAs("fakephoton_pdf_cat2_e.png")
# c1.SaveAs("fakephoton_pdf_cat2_m.png")

Data=ROOT.TFile.Open("/eos/user/s/shsong/HiggsDNA/root/cat2/Data_2017_e.root")
# Data=ROOT.TFile.Open("/eos/user/s/shsong/HiggsDNA/root/cat2/Data_2017_m.root")
Data_tree=Data.Get("cat2")
nevents=Data_tree.GetEntries()
new_weight=-999
weights=[]

minID=[]
maxID=[]
hasMaxLead=[]
originalminID=[]
passPhotonMVA90=[]
print(nevents)
for i in range(0,nevents):
    Data_tree.GetEntry(i)
    # weights.append(1)
    passPhotonMVA90.append(True)
    if(Data_tree.LeadPhoton_mvaID < Data_tree.SubleadPhoton_mvaID):
        hasleadIDmin=True
        original_Photon_MVA_min = Data_tree.LeadPhoton_mvaID
        Photon_MVA_max = Data_tree.SubleadPhoton_mvaID
    else:
        hasleadIDmin=False
        original_Photon_MVA_min = Data_tree.SubleadPhoton_mvaID
        Photon_MVA_max = Data_tree.LeadPhoton_mvaID
    originalminID.append(original_Photon_MVA_min)
    maxID.append(Photon_MVA_max)
    # weights.append(1)
    if(not (original_Photon_MVA_min<sideband_min and Photon_MVA_max>sideband_min)):
        new_weight=-999
        minID.append(-999)
        hasMaxLead.append(-999)
    else:

        if(hasleadIDmin):
            hasMaxLead.append(0)
            LeadPhoton_mvaID=photonIDPDF_fake.GetRandom(sideband_min,Photon_MVA_max)
            LeadPhoton_mvaID_WP90=True
            SubleadPhoton_mvaID_WP90=True
            is_passPhotonMVA90=True
            PhotonID_min=LeadPhoton_mvaID
        else:
            SubleadPhoton_mvaID=photonIDPDF_fake.GetRandom(sideband_min,Photon_MVA_max)
            PhotonID_min=SubleadPhoton_mvaID
            LeadPhoton_mvaID_WP90=True
            SubleadPhoton_mvaID_WP90=True
            is_passPhotonMVA90=True
            hasMaxLead.append(1)
        minID.append(PhotonID_min)
        new_weight = photonIDPDF_fake.Integral(sideband_min,Photon_MVA_max) / photonIDPDF_fake.Integral(-0.9,sideband_min);
        print(new_weight)
    weights.append(new_weight)
    if(i%100000==0):
        print("Read entry:",i,new_weight)

print(sum(weights))
d={"new_weight":weights,"minID":minID,"maxID":maxID,"originalminID":originalminID,"hasMaxLead":hasMaxLead,"passWP90":passPhotonMVA90}
import pandas
dataframe=pandas.DataFrame(d) 
# DataFile=awkward.from_parquet("/eos/user/s/shsong/HiggsDNA/parquet/cat2/Data_2017.parquet")

DataFile["Diphoton_maxID"]=dataframe.maxID
DataFile["Diphoton_minID"]=dataframe.minID
DataFile["originalminID"]=dataframe.originalminID
DataFile["weight_central"]=dataframe.new_weight
DataFile["is_passPhotonMVA90"]=dataframe.passWP90
DataFile=DataFile[DataFile.weight_central!=-999]
awkward.to_parquet(DataFile,"/eos/user/s/shsong/HiggsDNA/parquet/cat2/DatadrivenQCD_e.parquet")
parquet_to_root("/eos/user/s/shsong/HiggsDNA/parquet/cat2/DatadrivenQCD_e.parquet","/eos/user/s/shsong/HiggsDNA/root/cat2/DatadrivenQCD_e.root",treename="cat2",verbose=False)
# awkward.to_parquet(DataFile,"/eos/user/s/shsong/HiggsDNA/parquet/cat2/DatadrivenQCD_m.parquet")
# # parquet_to_root("/eos/user/s/shsong/HiggsDNA/parquet/cat2/DatadrivenQCD_m.parquet","/eos/user/s/shsong/HiggsDNA/root/cat2/DatadrivenQCD_m.root",treename="cat2",verbose=False)

# dataframe("data_weight.csv")