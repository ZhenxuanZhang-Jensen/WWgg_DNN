import ROOT
from array import array
import awkward 
import numpy
import pandas
from parquet_to_root import parquet_to_root
DataFile=awkward.from_parquet("/eos/user/s/shsong/HiggsDNA/data2017/merged_nominal.parquet")
DataFile=DataFile[DataFile.category==8]
DataFile['n_WP90']=awkward.sum(1*awkward.concatenate([awkward.unflatten(DataFile['SubleadPhoton_mvaID_WP90'],counts=1),awkward.unflatten(DataFile['LeadPhoton_mvaID_WP90'],counts=1)],axis=1),axis=1)

sideband_min=numpy.min(DataFile[DataFile.is_passPhotonMVA90].Diphoton_minID)
print(sideband_min)
GJets1=ROOT.TFile.Open("/eos/user/s/shsong/HiggsDNA/root/bkg/GJet.root")

GJets1_tree=GJets1.Get("Gjet")
GJets2_tree=GJets1.Get("Gjet")
EEcuts="(Diphoton_mass<115 || Diphoton_mass>135)*(minID_genPartFlav!=1)*(abs(minIDeta)>1.5)"
EBcuts="(Diphoton_mass<115 || Diphoton_mass>135)*(minID_genPartFlav!=1)*(abs(minIDeta)<1.5)"
h_minEEphotonID=ROOT.TH1F("h_minEEphotonID_gjet","h_minEEphotonID_gjet",19,-0.9,1)
h_minEBphotonID=ROOT.TH1F("h_minEBphotonID_gjet","h_minEBphotonID_gjet",19,-0.9,1)
GJets1_tree.Project("h_minEEphotonID_gjet","Diphoton_minID","weight_central*"+EEcuts)
GJets2_tree.Project("h_minEBphotonID_gjet","Diphoton_minID","weight_central*"+EBcuts)
EEphotonIDPDF_fake=ROOT.TF1("EEphotonIDPDF_fake","pol7",-0.9,1.)
EBphotonIDPDF_fake=ROOT.TF1("EBphotonIDPDF_fake","pol7",-0.9,1.)
h_minEEphotonID.Fit(EEphotonIDPDF_fake,"R")
h_minEBphotonID.Fit(EBphotonIDPDF_fake,"R")
c1=ROOT.TCanvas("c1","c1",600,800)
h_minEEphotonID.Draw("E1")
c1.SaveAs("fakeEEphoton_pdf_cat8.png")
c2=ROOT.TCanvas("c2","c2",600,800)
h_minEBphotonID.Draw("E1")
c2.SaveAs("fakeEBphoton_pdf_cat8.png")

Data=ROOT.TFile.Open("/eos/user/s/shsong/HiggsDNA/root/cat8/Data_2017.root")
Data_tree=Data.Get("cat8")
nevents=Data_tree.GetEntries()
new_weight=-999

newSubleadPhoton_mvaID=[]
newLeadPhoton_mvaID=[]
newSubleadPhoton_mvaID_WP90=[]
newLeadPhoton_mvaID_WP90=[]
newpassPhotonMVA90=[]
new_weight=[]
for i in range(0,nevents):
    Data_tree.GetEntry(i)
    # weights.append(1)
    if((Data_tree.LeadPhoton_mvaID_WP90 == True) & (Data_tree.SubleadPhoton_mvaID_WP90 ==True)):
        LeadPhoton_mvaID=-1
        SubleadPhoton_mvaID=-1
        LeadPhoton_mvaID_WP90=0
        SubleadPhoton_mvaID_WP90=0
        newpassPhotonMVA90.append(0)
        newSubleadPhoton_mvaID.append(SubleadPhoton_mvaID)
        newLeadPhoton_mvaID.append(LeadPhoton_mvaID)
        newSubleadPhoton_mvaID_WP90.append(SubleadPhoton_mvaID_WP90)
        newLeadPhoton_mvaID_WP90.append(LeadPhoton_mvaID_WP90)
        new_weight.append(-999)
    elif((Data_tree.LeadPhoton_mvaID_WP90 == False) & (Data_tree.SubleadPhoton_mvaID_WP90 ==False)):
        LeadPhoton_mvaID=-1
        SubleadPhoton_mvaID=-1
        LeadPhoton_mvaID_WP90=0
        SubleadPhoton_mvaID_WP90=0
        newpassPhotonMVA90.append(0)
        newSubleadPhoton_mvaID.append(SubleadPhoton_mvaID)
        newLeadPhoton_mvaID.append(LeadPhoton_mvaID)
        newSubleadPhoton_mvaID_WP90.append(SubleadPhoton_mvaID_WP90)
        newLeadPhoton_mvaID_WP90.append(LeadPhoton_mvaID_WP90)
        new_weight.append(-999)
    elif((Data_tree.LeadPhoton_mvaID_WP90 == False) & (Data_tree.SubleadPhoton_mvaID_WP90 == True)):
        if abs(Data_tree.LeadPhoton_eta)<1.5:#EB
            LeadPhoton_mvaID=EBphotonIDPDF_fake.GetRandom(-0.02,1)
            SubleadPhoton_mvaID=Data_tree.SubleadPhoton_mvaID
            LeadPhoton_mvaID_WP90=True
            SubleadPhoton_mvaID_WP90=Data_tree.SubleadPhoton_mvaID_WP90
            passPhotonMVA90=1
            if abs(Data_tree.SubleadPhoton_eta)<1.5:#EB
                shiftweight=EBphotonIDPDF_fake.Integral(-0.02,Data_tree.SubleadPhoton_mvaID) / EBphotonIDPDF_fake.Integral(-0.9,-0.02);

            if abs(Data_tree.SubleadPhoton_eta)>1.5:
                shiftweight=EEphotonIDPDF_fake.Integral(-0.26,Data_tree.SubleadPhoton_mvaID) / EEphotonIDPDF_fake.Integral(-0.9,-0.26);
        if (abs(Data_tree.LeadPhoton_eta)>1.5):#EE
            LeadPhoton_mvaID=EEphotonIDPDF_fake.GetRandom(-0.26,1)
            SubleadPhoton_mvaID=Data_tree.SubleadPhoton_mvaID
            LeadPhoton_mvaID_WP90=True
            SubleadPhoton_mvaID_WP90=Data_tree.SubleadPhoton_mvaID_WP90
            passPhotonMVA90=1
            if abs(Data_tree.SubleadPhoton_eta)<1.5:#EB
                shiftweight=EBphotonIDPDF_fake.Integral(-0.02,Data_tree.SubleadPhoton_mvaID) / EBphotonIDPDF_fake.Integral(-0.9,-0.02);
            if abs(Data_tree.SubleadPhoton_eta)>1.5:#EE
                shiftweight=EEphotonIDPDF_fake.Integral(-0.26,Data_tree.SubleadPhoton_mvaID) / EEphotonIDPDF_fake.Integral(-0.9,-0.26);
        else:
            LeadPhoton_mvaID=-1
            SubleadPhoton_mvaID=-1
            LeadPhoton_mvaID_WP90=0
            SubleadPhoton_mvaID_WP90=0
            passPhotonMVA90=0
            shiftweight==-999
        newpassPhotonMVA90.append(passPhotonMVA90)
        newSubleadPhoton_mvaID.append(SubleadPhoton_mvaID)
        newLeadPhoton_mvaID.append(LeadPhoton_mvaID)
        newSubleadPhoton_mvaID_WP90.append(SubleadPhoton_mvaID_WP90)
        newLeadPhoton_mvaID_WP90.append(LeadPhoton_mvaID_WP90)
        new_weight.append(shiftweight)
    elif((Data_tree.LeadPhoton_mvaID_WP90 == True) &(Data_tree.SubleadPhoton_mvaID_WP90 == False)):
        if abs(Data_tree.SubleadPhoton_eta)<1.5:#EB
            SubleadPhoton_mvaID=EBphotonIDPDF_fake.GetRandom(-0.02,1)
            LeadPhoton_mvaID=Data_tree.LeadPhoton_mvaID
            SubleadPhoton_mvaID_WP90=True
            LeadPhoton_mvaID_WP90=Data_tree.LeadPhoton_mvaID_WP90
            if abs(Data_tree.LeadPhoton_eta)<1.5:#EB
                shiftweight=EBphotonIDPDF_fake.Integral(-0.02,Data_tree.LeadPhoton_mvaID) / EBphotonIDPDF_fake.Integral(-0.9,-0.02);
                # shiftweight= EBphotonIDPDF_fake.Integral(-0.9,-0.02)/EBphotonIDPDF_fake.Integral(-0.02,Data_tree.LeadPhoton_mvaID) 
            if abs(Data_tree.LeadPhoton_eta)>1.5:
                shiftweight=EEphotonIDPDF_fake.Integral(-0.26,Data_tree.LeadPhoton_mvaID) / EEphotonIDPDF_fake.Integral(-0.9,-0.26);
                # shiftweight= EEphotonIDPDF_fake.Integral(-0.9,-0.26)/EEphotonIDPDF_fake.Integral(-0.26,Data_tree.LeadPhoton_mvaID)

            # new_weight = EBphotonIDPDF_fake.Integral(-0.02,1) / EBphotonIDPDF_fake.Integral(-0.9,-0.02);
        if (abs(Data_tree.SubleadPhoton_eta)>1.5):#EE
            SubleadPhoton_mvaID=EEphotonIDPDF_fake.GetRandom(-0.26,1)
            LeadPhoton_mvaID=Data_tree.LeadPhoton_mvaID
            SubleadPhoton_mvaID_WP90=True
            LeadPhoton_mvaID_WP90=Data_tree.LeadPhoton_mvaID_WP90
            if abs(Data_tree.LeadPhoton_eta)<1.5:#EB
                shiftweight=EBphotonIDPDF_fake.Integral(-0.02,Data_tree.LeadPhoton_mvaID) / EBphotonIDPDF_fake.Integral(-0.9,-0.02);
                # shiftweight= EBphotonIDPDF_fake.Integral(-0.9,-0.02)/EBphotonIDPDF_fake.Integral(-0.02,Data_tree.LeadPhoton_mvaID) 
            if abs(Data_tree.LeadPhoton_eta)>1.5:
                shiftweight=EEphotonIDPDF_fake.Integral(-0.26,Data_tree.LeadPhoton_mvaID) / EEphotonIDPDF_fake.Integral(-0.9,-0.26);
                # shiftweight= EEphotonIDPDF_fake.Integral(-0.9,-0.26)/EEphotonIDPDF_fake.Integral(-0.26,Data_tree.LeadPhoton_mvaID)

        else:
            LeadPhoton_mvaID=-1
            SubleadPhoton_mvaID=-1
            LeadPhoton_mvaID_WP90=0
            SubleadPhoton_mvaID_WP90=0
            passPhotonMVA90=0
            shiftweight=-999
        newSubleadPhoton_mvaID.append(SubleadPhoton_mvaID)
        newLeadPhoton_mvaID.append(LeadPhoton_mvaID)
        newSubleadPhoton_mvaID_WP90.append(SubleadPhoton_mvaID_WP90)
        newLeadPhoton_mvaID_WP90.append(LeadPhoton_mvaID_WP90)
        newpassPhotonMVA90.append(passPhotonMVA90)
        new_weight.append(shiftweight)

d={"is_passPhotonMVA90":newpassPhotonMVA90,"newSubleadPhoton_mvaID":newSubleadPhoton_mvaID,"newLeadPhoton_mvaID":newLeadPhoton_mvaID,"newSubleadPhoton_mvaID_WP90":newSubleadPhoton_mvaID_WP90,"newLeadPhoton_mvaID_WP90":newLeadPhoton_mvaID_WP90,"new_weight":new_weight}
dataframe=pandas.DataFrame(d) 
DataFile["Diphoton_maxID"]=numpy.maximum(dataframe.newLeadPhoton_mvaID, dataframe.newSubleadPhoton_mvaID)
DataFile["Diphoton_minID"]=numpy.minimum(dataframe.newLeadPhoton_mvaID, dataframe.newSubleadPhoton_mvaID)
DataFile["LeadPhoton_mvaID"]=dataframe.newLeadPhoton_mvaID
DataFile["SubleadPhoton_mvaID"]=dataframe.newSubleadPhoton_mvaID
DataFile["is_passPhotonMVA90"]=awkward.ones_like(DataFile.is_passPhotonMVA90)
DataFile["LeadPhoton_mvaID_WP90"]=awkward.ones_like(DataFile.LeadPhoton_mvaID_WP90)
DataFile["SubleadPhoton_mvaID_WP90"]=awkward.ones_like(DataFile.LeadPhoton_mvaID_WP90)
# DataFileshift["weight_central"]=dataframe.new_weight
DataFile=DataFile[DataFile.n_WP90==1]
# DataFileshift=DataFileshift[DataFileshift.n_WP90==1]


# awkward.to_parquet(DataFileshift,"/eos/user/s/shsong/HiggsDNA/parquet/cat8/DatadrivenQCDshift.parquet")

awkward.to_parquet(DataFile,"/eos/user/s/shsong/HiggsDNA/parquet/cat8/DatadrivenQCD.parquet")

parquet_to_root("/eos/user/s/shsong/HiggsDNA/parquet/cat8/DatadrivenQCD.parquet","/eos/user/s/shsong/HiggsDNA/root/cat8/DatadrivenQCD.root",treename="cat8",verbose=False)
# parquet_to_root("/eos/user/s/shsong/HiggsDNA/parquet/cat8/DatadrivenQCDshift.parquet","/eos/user/s/shsong/HiggsDNA/root/cat8/DatadrivenQCDshift.root",treename="cat8",verbose=False)

# dataframe("data_weight.csv")