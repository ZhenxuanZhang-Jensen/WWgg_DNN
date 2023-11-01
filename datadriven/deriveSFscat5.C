
class chisquare 
{

public:
  chisquare(TH1F* data1, TH1F* constMC1, TH1F* diphotonMC1, TH1F* datadrivenQCD1,
	    TH1F* data2, TH1F* constMC2, TH1F* diphotonMC2, TH1F* datadrivenQCD2)
  {
    data1_ = data1; 
    constMC1_ = constMC1; 
    diphotonMC1_ = diphotonMC1; 
    datadrivenQCD1_ = datadrivenQCD1;
    data2_ = data2; 
    constMC2_ = constMC2; 
    diphotonMC2_ = diphotonMC2; 
    datadrivenQCD2_ = datadrivenQCD2;
    SFdiphoton_ = 1.;
    SFdatadrivenQCD_ = 1.;
  }
  double operator()( double* SFs, double *p)
  {
    SFdiphoton_ = SFs[0];
    SFdatadrivenQCD_ = SFs[1];
    // cout<<"SFdiphoton_:"<<SFdiphoton_<<endl;
    // cout<<"SFdatadrivenQCD_:"<<SFdatadrivenQCD_<<endl;
    double chisquarevalue = 0.;
    for(int ibin=1; ibin<=data1_->GetXaxis()->GetNbins(); ++ibin) {
      double Nev_data1 = data1_->GetBinContent(ibin);
      double Nev_constMC1 = constMC1_->GetBinContent(ibin);
      double Nev_diphotonMC1 = SFdiphoton_ * diphotonMC1_->GetBinContent(ibin);
      double Nev_datadrivenQCDMC1 = SFdatadrivenQCD_ * datadrivenQCD1_->GetBinContent(ibin);
      double Nev_exp1 = Nev_constMC1+Nev_diphotonMC1+Nev_datadrivenQCDMC1; 
      // cout<<"Nev_data1:"<<Nev_data1<<endl;
      // cout<<"Nev_exp1:"<<Nev_exp1<<endl;
    
      if (Nev_data1 == 0) {
        Nev_data1 = 1e-6; // 或者其他很小的值
      }
      else{
        chisquarevalue += (Nev_data1-Nev_exp1)*(Nev_data1-Nev_exp1) / Nev_data1;}
      // cout<<"chisquarevaluemin:"<<chisquarevalue<<endl;
    }
    for(int ibin=1; ibin<=data2_->GetXaxis()->GetNbins(); ++ibin) {
      double Nev_data2 = data2_->GetBinContent(ibin);
      double Nev_constMC2 = constMC2_->GetBinContent(ibin);
      double Nev_diphotonMC2 = SFdiphoton_ * diphotonMC2_->GetBinContent(ibin);
      double Nev_datadrivenQCDMC2 = SFdatadrivenQCD_ * datadrivenQCD2_->GetBinContent(ibin);
      double Nev_exp2 = Nev_constMC2+Nev_diphotonMC2+Nev_datadrivenQCDMC2; 


      if (Nev_data2 == 0) {
        Nev_data2 = 1e-6; // 或者其他很小的值
      }
      else {  
        chisquarevalue += (Nev_data2-Nev_exp2)*(Nev_data2-Nev_exp2) / Nev_data2;
        // cout<<"chisquarevalue:"<<chisquarevalue<<endl;
      }
      // cout<<"Nev_data2:"<<Nev_data2<<endl;
      // cout<<"Nev_exp2:"<<Nev_exp2<<endl;
    }
  

    // cout<<"chisquarevaluemax:"<<chisquarevalue<<endl;
    return chisquarevalue;
  }
  
private:
  TH1F *data1_, *constMC1_, *diphotonMC1_, *datadrivenQCD1_;
  TH1F *data2_, *constMC2_, *diphotonMC2_, *datadrivenQCD2_;
  double SFdiphoton_, SFdatadrivenQCD_;
};

void deriveSFscat5()
{

  string common_selection = "(Diphoton_mass<115 || Diphoton_mass>135)*(is_passPhotonMVA90)*(category==5)";
  map<string, vector<string> > filenames_map;
  map<string, vector<string> > treenames_map;
  map<string, vector<float> > lumis_map;
  filenames_map["data"] = { vector<string>{
      "/eos/user/s/shsong/HiggsDNA/root/cat5/Data_2017.root"
    }};
  filenames_map["qcd"] = { vector<string>{
      "/eos/user/s/shsong/HiggsDNA/root/cat5/DatadrivenQCD.root"
    }};
  filenames_map["diphoton"] = { vector<string>{
      "/eos/user/s/shsong/HiggsDNA/root/cat5/DiPhotonJetsBox.root"
    }};
  treenames_map["data"] = { vector<string>{
      "cat5"
    }};
  treenames_map["qcd"] = { vector<string>{
      "cat5"
    }};
  treenames_map["diphoton"] = { vector<string>{
      "cat5"
    }};
  lumis_map["data"] = {vector<float>{
    1
  }};
  lumis_map["qcd"] = {vector<float>{
    1
  }};
  lumis_map["diphoton"] = {vector<float>{
    1}};
  cout<<"reading all root files"<<endl;
  //Get min and max photon ID distribution
  map<string,TH1F*> h_minphotonID;
  map<string,TH1F*> h_maxphotonID;
  for( auto filenameitr : filenames_map) {
    auto samplename = filenameitr.first;
    auto filenames = filenameitr.second;
    auto treenames = treenames_map[samplename];
    auto lumis = lumis_map[samplename];
    h_minphotonID[samplename] = new TH1F(Form("h_minphotonID_%s",samplename.c_str()),
					 Form("h_minphotonID_%s",samplename.c_str()),
					 12,-0.2,1);
    h_maxphotonID[samplename] = new TH1F(Form("h_maxphotonID_%s",samplename.c_str()),
					 Form("h_maxphotonID_%s",samplename.c_str()),
					 12,-0.2,1);
    for( unsigned ifile=0; ifile<filenames.size(); ++ifile) {
      TChain* ch = new TChain();
      ch->Add( Form("%s/%s",filenames[ifile].c_str(),treenames[ifile].c_str()) );
      ch->Draw( Form("Diphoton_minID >>+ h_minphotonID_%s",samplename.c_str()),
		Form("(weight_central)*(%f)*(%s)",lumis[ifile],common_selection.c_str()),
		"goff");
      ch->Draw( Form("Diphoton_maxID >>+ h_maxphotonID_%s",samplename.c_str()),
		Form("(weight_central)*(%f)*(%s)",lumis[ifile],common_selection.c_str()),
		"goff");
    }
  }

  h_minphotonID["otherMC"] = new TH1F("constMC1","constMC1",12,-0.2,1);  
  h_minphotonID["MCtot"] = new TH1F("MCtot1","MCtot1",12,-0.2,1);
  h_minphotonID["scaledMCtot"] = new TH1F("MCtot1_scaled","MCtot1_scaled",12,-0.2,1);
  h_maxphotonID["otherMC"] = new TH1F("constMC2","constMC2",12,-0.2,1); 
  h_maxphotonID["MCtot"] = new TH1F("MCtot2","MCtot2",12,-0.2,1);
  h_maxphotonID["scaledMCtot"] = new TH1F("MCtot2_scaled","MCtot2_scaled",12,-0.2,1);

  TCanvas* c1 = new TCanvas();
  h_minphotonID["MCtot"]->Add(h_minphotonID["otherMC"]);
  h_minphotonID["MCtot"]->Add(h_minphotonID["diphoton"]);
  h_minphotonID["MCtot"]->Add(h_minphotonID["qcd"]);
  h_minphotonID["MCtot"]->Draw("hist");
  h_minphotonID["data"]->SetMarkerStyle(20);
  h_minphotonID["data"]->Draw("E1 same");
  h_minphotonID["MCtot"]->GetYaxis()->SetRangeUser(0., 1.3*TMath::Max(h_minphotonID["MCtot"]->GetMaximum(),h_minphotonID["data"]->GetMaximum()) );
  cout<<"Data minID:"<<h_minphotonID["data"]->Integral()<<endl;
  cout<<"QCD minID:"<<h_minphotonID["qcd"]->Integral()<<endl;
  cout<<"DiPhoton minID:"<<h_minphotonID["diphoton"]->Integral()<<endl;

  c1->SaveAs("Diphoton_minID_cat5.png");

  TCanvas* c12 = new TCanvas();
  h_maxphotonID["MCtot"]->Add(h_maxphotonID["otherMC"]);
  h_maxphotonID["MCtot"]->Add(h_maxphotonID["diphoton"]);
  h_maxphotonID["MCtot"]->Add(h_maxphotonID["qcd"]);
  h_maxphotonID["MCtot"]->Draw("hist");
  h_maxphotonID["data"]->SetMarkerStyle(20);
  h_maxphotonID["data"]->Draw("E1 same");
  h_maxphotonID["MCtot"]->GetYaxis()->SetRangeUser(0., 1.3*TMath::Max(h_maxphotonID["MCtot"]->GetMaximum(),h_maxphotonID["data"]->GetMaximum()) );
  cout<<"Data maxID:"<<h_maxphotonID["data"]->Integral()<<endl;
  cout<<"QCD maxID:"<<h_maxphotonID["qcd"]->Integral()<<endl;
  cout<<"DiPhoton maxID:"<<h_maxphotonID["diphoton"]->Integral()<<endl;
  chisquare chisquareobj(h_minphotonID["data"], h_minphotonID["otherMC"], h_minphotonID["diphoton"], h_minphotonID["qcd"], 
			 h_maxphotonID["data"], h_maxphotonID["otherMC"], h_maxphotonID["diphoton"], h_maxphotonID["qcd"]);
  cout<<"getting all chisquareobj"<<endl;
  TF2 *f = new TF2("chi2",chisquareobj,0.0001,10.,0.0001,10.,0);
  c12->SaveAs("Diphoton_maxID_cat5.png");

  double SFdiphoton,SFdatadrivenQCD;
  double chi2 = f->GetMinimumXY(SFdiphoton,SFdatadrivenQCD);
  cout<<"observed: "<<SFdiphoton<<" "<<SFdatadrivenQCD<<" chi2="<<chi2<<endl;

  TCanvas* c2 = new TCanvas();
  h_minphotonID["diphoton"]->Scale(SFdiphoton);
  h_minphotonID["qcd"]->Scale(SFdatadrivenQCD);
  h_minphotonID["scaledMCtot"]->Add(h_minphotonID["otherMC"]);
  h_minphotonID["scaledMCtot"]->Add(h_minphotonID["diphoton"]);
  h_minphotonID["scaledMCtot"]->Add(h_minphotonID["qcd"]);
  h_minphotonID["scaledMCtot"]->Draw("hist");
  h_minphotonID["data"]->Draw("E1 same");
  h_minphotonID["scaledMCtot"]->GetYaxis()->SetRangeUser(0., 1.3*TMath::Max(h_minphotonID["scaledMCtot"]->GetMaximum(),h_minphotonID["data"]->GetMaximum()) );

  c2->SaveAs("minID_scale_cat5.png");
  TCanvas* c22 = new TCanvas();

  h_maxphotonID["diphoton"]->Scale(SFdiphoton);
  h_maxphotonID["qcd"]->Scale(SFdatadrivenQCD);
  h_maxphotonID["scaledMCtot"]->Add(h_maxphotonID["otherMC"]);
  h_maxphotonID["scaledMCtot"]->Add(h_maxphotonID["diphoton"]);
  h_maxphotonID["scaledMCtot"]->Add(h_maxphotonID["qcd"]);
  h_maxphotonID["scaledMCtot"]->Draw("hist");
  h_maxphotonID["data"]->Draw("E1 same");
  cout<<"Data:"<<h_maxphotonID["data"]->Integral()<<endl;
  cout<<"QCD:"<<h_maxphotonID["qcd"]->Integral()<<endl;
  cout<<"DiPhoton:"<<h_maxphotonID["diphoton"]->Integral()<<endl;
  h_maxphotonID["scaledMCtot"]->GetYaxis()->SetRangeUser(0., 1.3*TMath::Max(h_maxphotonID["scaledMCtot"]->GetMaximum(),h_maxphotonID["data"]->GetMaximum()) );


  c22->SaveAs("maxID_scale_cat5.png");
}

