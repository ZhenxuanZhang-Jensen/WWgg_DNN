import ROOT
from array import array
import awkward 
import numpy
import pandas
from parquet_to_root import parquet_to_root
DataFile=awkward.from_parquet("/eos/user/s/shsong/HiggsDNA/data2017/merged_nominal.parquet")
DataFile=DataFile[DataFile.category==8]
sideband_min=numpy.min(DataFile[DataFile.is_passPhotonMVA90].Diphoton_minID)
print(sideband_min)
GJets1=ROOT.TFile.Open("/eos/user/s/shsong/HiggsDNA/root/bkg/GJet.root")

GJets1_tree=GJets1.Get("Gjet")
cuts="(Diphoton_mass<115 || Diphoton_mass>135)*(minID_genPartFlav!=1)*(category==8)"
h_minphotonID=ROOT.TH1F("h_minphotonID_gjet","h_minphotonID_gjet",19,-0.9,1)
GJets1_tree.Project("h_minphotonID_gjet","Diphoton_minID","weight_central*"+cuts)
photonIDPDF_fake=ROOT.TF1("photonIDPDF_fake","pol7",-0.9,1.)
h_minphotonID.Fit(photonIDPDF_fake,"R")
# c1=ROOT.TCanvas("c1","c1",600,800)
# h_minphotonID.Draw("E1")
# c1.SaveAs("fakephoton_pdf_cat8.png")

Data=ROOT.TFile.Open("/eos/user/s/shsong/HiggsDNA/root/cat8/Data_2017.root")
Data_tree=Data.Get("cat8")
nevents=Data_tree.GetEntries()
new_weight=-999
weights=[]

minID=[]
maxID=[]
hasMaxLead=[]
originalminID=[]
passPhotonMVA90=[]
print(nevents)
j=0
for i in range(0,nevents):
    Data_tree.GetEntry(i)
    # weights.append(1)
    is_passPhotonMVA90=False
    if(Data_tree.LeadPhoton_mvaID < Data_tree.SubleadPhoton_mvaID):
        hasleadIDmin=True
        original_Photon_MVA_min = Data_tree.LeadPhoton_mvaID
        Photon_MVA_max = Data_tree.SubleadPhoton_mvaID
        original_min_MVA90 = Data_tree.LeadPhoton_mvaID_WP90
        Photon_max_WP90 = Data_tree.SubleadPhoton_mvaID_WP90
    else:
        hasleadIDmin=False
        original_Photon_MVA_min = Data_tree.SubleadPhoton_mvaID
        Photon_MVA_max = Data_tree.LeadPhoton_mvaID
        original_min_MVA90 = Data_tree.SubleadPhoton_mvaID_WP90
        Photon_max_WP90 = Data_tree.LeadPhoton_mvaID_WP90
    originalminID.append(original_Photon_MVA_min)
    maxID.append(Photon_MVA_max)
    # weights.append(1)
    if(not (original_min_MVA90==False and Photon_max_WP90==True)):
        new_weight=-999
        minID.append(-999)
        hasMaxLead.append(-999)
        passPhotonMVA90.append(is_passPhotonMVA90)
    else:
        if (sideband_min>=Photon_MVA_max):
            print(sideband_min)
            print(Photon_MVA_max)
            j=j+1
            print(j)
            new_weight=-999
            minID.append(-999)
            hasMaxLead.append(-999)
            passPhotonMVA90.append(is_passPhotonMVA90)
        else:
            is_passPhotonMVA90=True
            passPhotonMVA90.append(is_passPhotonMVA90)
            if(hasleadIDmin):
                hasMaxLead.append(0)
                LeadPhoton_mvaID=photonIDPDF_fake.GetRandom(sideband_min,Photon_MVA_max)
                PhotonID_min=LeadPhoton_mvaID
            else:
                SubleadPhoton_mvaID=photonIDPDF_fake.GetRandom(sideband_min,Photon_MVA_max)
                PhotonID_min=SubleadPhoton_mvaID
                hasMaxLead.append(1)
            minID.append(PhotonID_min)
            new_weight = photonIDPDF_fake.Integral(sideband_min,Photon_MVA_max) / photonIDPDF_fake.Integral(-0.9,sideband_min);
    weights.append(new_weight)
    if(i%100000==0):
        print("Read entry:",i,new_weight)

print(sum(weights))
print(nevents)
d={"new_weight":weights,"minID":minID,"maxID":maxID,"originalminID":originalminID,"hasMaxLead":hasMaxLead,"passWP90":passPhotonMVA90}
dataframe=pandas.DataFrame(d) 
# DataFile=awkward.from_parquet("/eos/user/s/shsong/HiggsDNA/parquet/cat8/Data_2017.parquet")

DataFile["Diphoton_maxID"]=dataframe.maxID
DataFile["Diphoton_minID"]=dataframe.minID
DataFile["originalminID"]=dataframe.originalminID
DataFile["weight_central"]=dataframe.new_weight
DataFile["is_passPhotonMVA90"]=dataframe.passWP90
DataFile=DataFile[DataFile.weight_central!=-999]
print(len(DataFile))
awkward.to_parquet(DataFile,"/eos/user/s/shsong/HiggsDNA/parquet/cat8/DatadrivenQCD.parquet")

parquet_to_root("/eos/user/s/shsong/HiggsDNA/parquet/cat8/DatadrivenQCD.parquet","/eos/user/s/shsong/HiggsDNA/root/cat8/DatadrivenQCD.root",treename="cat8",verbose=False)

# dataframe("data_weight.csv")