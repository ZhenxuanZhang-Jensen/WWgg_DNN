 //#include "setTDRStyle.C"
#include "TMath.h"
//#include "TROOT.h"
#include <TH1D.h>
#include <TH1D.h>
#include <TH1.h>
#include <TProfile.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLeafF.h>
#include <TChain.h>
#include <TFile.h>
#include "TSystem.h"
#include <TChain.h>
#include "TSystem.h"
#include <TString.h>
#include <iostream>
#include <vector>
#include <TPostScript.h>
#include <iostream>
#include <iomanip>  //for precision

//==============
const int debug=1;

//after training
// const TString Inputgg80toInfFile ="/eos/user/s/shsong/combined_WWgg/datadriven/cat7/DiphotonJetbox_reweight.root";
const TString Inputgg80toInfFile ="/eos/user/s/shsong/HHWWgg/root/cat7/DiPhotonJetsBox.root";
const TString TreeNamegg80toInf = "cat7";

// const TString InputQCD40InfFile = "/eos/user/s/shsong/combined_WWgg/datadriven/cat7/DatadrivenQCD_reweight.root";
const TString InputQCD40InfFile = "/eos/user/s/shsong/HHWWgg/root/cat7/DatadrivenQCD.root";
const TString TreeNameQCD40Inf = "cat7";

// const TString InputDataFile = "/eos/user/s/shsong/combined_WWgg/datadriven/cat7/Data_new.root";
const TString InputDataFile = "/eos/user/s/shsong/HHWWgg/root/cat7/Data_2017.root";
const TString TreeNameData = "cat7";

// const TString InputSigFile="/eos/user/s/shsong/combined_WWgg/DNN/cat7/m600.root";
// const TString InputSigFile="/eos/user/s/shsong/combined_WWgg/datadriven/cat7/m600.root";
const TString InputSigFile="/eos/user/s/shsong/HHWWgg/root/cat7/FH250.root";
const TString TreeNameSig = "cat7";


// const TString InputWGGFile = "/eos/user/s/shsong/combined_WWgg/root/bkg/WGGJets_2017_new.root";
//const TString TreeNameWGG = "WGGJets";


// const string OutputPlotDir = "/eos/user/s/shsong/www/cat3dd/";
const string OutputPlotDir = "/eos/user/s/shsong/www/combined_Data_MC_cat7dd";
// const string OutputPlotDir = "/eos/user/s/shsong/www/combined_Data_MC_cat7_electronddv2";
// const string OutputPlotDir = "/eos/user/s/shsong/www/cat3dd_electron/";


// const string Preselections="( Diphoton_mass <= 115. || Diphoton_mass >= 135.)&& (category==3)&&(Diphoton_minID>-0.7)&&(electron_pt>-999)";
const string Preselections="( Diphoton_mass <= 115. || Diphoton_mass >= 135.)&&(Diphoton_minID>-0.7)";
// const string Preselections="( Diphoton_mass <= 115. || Diphoton_mass >= 135.)&& category==2&&(Diphoton_minID>-0.7)&&(Diphoton_maxID>-0.7)";//&&(dnn_score<0.91)";
// "weight*41.5"; // add a new weight to scale diphoton bdt score >-0.85
const string MCWeight = "weight_central";

// */
//====================



///////////////////////////////////////

void DrawMyPlots(string Object, string Selections,  string XTitle, string YUnit, string PlotName, int BinTotal, double BinXLow, double BinXHig, int LegendLR=0, int IfLogY=0, int IfStatErr=0){


  TCanvas *c1 = new TCanvas("reconstruction1","reconstruction1");
  c1->SetFillColor(0);

  // TLegend *legend;

  // if(IfStatErr==1){
  //   if(LegendLR==0) legend = new TLegend(0.65,0.67,0.89,0.9); 
  //   else if (LegendLR==-1) legend = new TLegend(0.45,0.67,0.69,0.9); //Middle
  //   else legend = new TLegend(0.2,0.67,0.44,0.9);
  // }else{
  //   if(LegendLR==0) legend = new TLegend(0.65,0.7,0.89,0.9); //(0.7,0.75,0.9,0.9);
  //   else if (LegendLR==-1) legend = new TLegend(0.45,0.7,0.69,0.9); //Middle
  //   else legend = new TLegend(0.2,0.7,0.44,0.9);  
  // }
  // legend->SetFillColor(0);

  //====add root file
  TChain *Data_Tree=new TChain(TreeNameData);
  TChain *MCgg80toInf_Tree=new TChain(TreeNamegg80toInf);
  
  // TChain *MCTTGG0Jets_Tree=new TChain(TreeNameTTGG0Jets);

  // // TChain *MCQCD30Inf_Tree=new TChain(TreeNameQCD30Inf);
  TChain *MCQCD40Inf_Tree=new TChain(TreeNameQCD40Inf);
  TChain *MCsig_Tree=new TChain(TreeNameSig) ; 


  //====================
  Data_Tree->Add(InputDataFile);
  MCgg80toInf_Tree->Add(Inputgg80toInfFile);
 
  // // MCW1Jets_Tree->Add(InputW1JetsFile);

  // MCTTGG0Jets_Tree->Add(InputTTGG0JetsFile);
  MCsig_Tree->Add(InputSigFile);
  // // MCQCD30Inf_Tree->Add(InputQCD30InfFile);
  MCQCD40Inf_Tree->Add(InputQCD40InfFile);  

  
  // MCWGG_Tree->Add(InputWGGFile);



  //=========entries================
  int entries_Data = Data_Tree->GetEntries();
  if(debug==1) cout <<" nEntries_Data = "<<entries_Data<<endl;

  //int entries_MC = MC_Tree->GetEntries();
  //if(debug==1) cout <<" nEntries_MC = "<<entries_MC<<endl;

  c1->cd();

  //=================
  char *myLimits= new char[100];
  sprintf(myLimits,"(%d,%f,%f)",BinTotal,BinXLow,BinXHig);
  TString Taolimits(myLimits);

  cout<<"selections -- "<<Selections<<endl;
  //====data=======
  TString variable_Data = Object + ">>Histo_Data_temp" + Taolimits;
  Data_Tree->Draw(variable_Data, Selections.c_str());
  TH1D *h_data = (TH1D*)gDirectory->Get("Histo_Data_temp");
  h_data->SetTitle("");
  c1->Clear();

  double Ntot_Data=h_data->Integral();
  if( debug==1 ) cout<<" N_Data= "<<Ntot_Data<<endl;

  Double_t scale_Data = 1.0/Ntot_Data;
  h_data->Sumw2();
  //h_data->Scale(scale_Data);
  const string MCWeight1 = "1.27936*weight_central";
  const string MCWeight2 = "1.06415*weight_central";  


  //---MC---
  string MCSelections1 = MCWeight1 + "*(" + Selections + ")";
  string MCSelections2 = MCWeight2+ "*(" + Selections + ")";

  std::cout << "MCSelections"<<MCSelections1<< std::endl;
  const string Sig_weight = "weight_central"; // add a new weight to scale diphoton bdt score >-0.85
  cout<<"MC selections -- "<<MCSelections1<<endl;

  TString variable_MCgg80toInf = Object + ">>Histo_MCgg80toInf_temp" + Taolimits;
  MCgg80toInf_Tree->Draw(variable_MCgg80toInf,MCSelections1.c_str());
  TH1D *h_MCgg80toInf = (TH1D*)gDirectory->Get("Histo_MCgg80toInf_temp");
  c1->Clear();
  


  TString variable_MCQCD40Inf = Object + ">>Histo_MCQCD40Inf_temp" + Taolimits;
  MCQCD40Inf_Tree->Draw(variable_MCQCD40Inf,MCSelections2.c_str());
  TH1D *h_MCQCD40Inf = (TH1D*)gDirectory->Get("Histo_MCQCD40Inf_temp");
  c1->Clear();


 TString variable_MCsig = Object + ">>Histo_MCsig_temp" + Taolimits;
  MCsig_Tree->Draw(variable_MCsig, Sig_weight.c_str());
  TH1D *h_MCsig = (TH1D*)gDirectory->Get("Histo_MCsig_temp");
  h_MCsig->Scale(10000);
  std::cout << "MCsig entries=" << h_MCsig->Integral() << std::endl;
  c1->Clear();



  h_MCgg80toInf->SetLineColor(kOrange);
  h_MCgg80toInf->SetFillColor(kOrange);
  // h_MCgg80toInf->SetFillStyle(3004);
  h_MCgg80toInf->SetLineStyle(1);
  h_MCgg80toInf->SetLineWidth(2);




  h_MCQCD40Inf->SetLineColor(kGreen+2);
  h_MCQCD40Inf->SetFillColor(kGreen+2);
  //hpf->SetFillStyle(3005);
  h_MCQCD40Inf->SetLineStyle(1);
  h_MCQCD40Inf->SetLineWidth(2);

  // cout << "scale factor:" << scale_MC << endl;
  // hs->Scale(scale_MC);
  //h_MC->Scale(MCXSweight);

  TH1D *h_MC=new TH1D("h_MC","",BinTotal,BinXLow,BinXHig);
  h_MC->Sumw2();
  // h_MC->Add(h_MCgg40to80,1.0);
  h_MC->Add(h_MCgg80toInf,1.0);
  
  // h_MC->Add(h_MCTTGG0Jets,1.0);
  
  h_MC->Add(h_MCQCD40Inf,1.0);
  double Ntot_MC=h_MC->Integral();
  if( debug==1 ) cout<<" N_MC= "<<Ntot_MC<<endl;
  Double_t scale_MC = Ntot_Data*1.0/Ntot_MC;
  cout << "Data/MC discrepancy = " <<scale_MC<<endl;
  std::cout << "MCDiphotonJetbox80toInf entries=" << h_MCgg80toInf->Integral() << std::endl;
  // std::cout << "MCSig entries=" << h_MCsig->Integral() << std::endl;


  // // std::cout << "MCQCD30Inf entries=" << h_MCQCD30Inf->Integral()<<"  "<< (h_MCQCD30Inf->Integral())/Ntot_MC*100<<"%" << std::endl;
  std::cout << "MCQCD40Inf entries=" << h_MCQCD40Inf->Integral()<< std::endl;
  scale_MC = 1;  
  h_MC->Sumw2();
  h_MCgg80toInf->Scale(scale_MC);
  
  
  h_MCQCD40Inf->Scale(scale_MC);
 
  THStack *h_signal=new THStack("hs1","");
  h_signal->Add(h_MCsig);

  THStack *hs = new THStack("hs","");

  // hs->Add(h_MCgg40to80);
  hs->Add(h_MCgg80toInf);
  
  // hs->Add(h_MCTTGG0Jets);
  
  // hs->Add(h_MCQCD30Inf);
  hs->Add(h_MCQCD40Inf);
  
  double Chi2=0.;
  for(int ibin=0; ibin<BinTotal; ibin++){
    double Nd = h_data->GetBinContent(ibin+1);
    double Nm = h_MC->GetBinContent(ibin+1);
    double NmErr = h_MC->GetBinError(ibin+1);
    
    Chi2 += fabs(NmErr)>1e-9?(Nm-Nd)*(Nm-Nd)*1.0/(NmErr*NmErr):0.0;
  }

  cout<<" chi2 = "<<Chi2<<endl;

  //Stat Err

  TH1D *htot=new TH1D("htot","",BinTotal,BinXLow,BinXHig);
  TH1D *htot_Norm=new TH1D("htot_Norm","",BinTotal,BinXLow,BinXHig);
  // if(IfStatErr==1){
  //   htot->Add(h_MC,1.0);
  //   htot->Sumw2();
  //   //double Ntot_MC=htot->Integral();
  //   //float scale = Ntot_Data*1.0/Ntot_MC;
  //   //htot->Scale(scale);

  //   }
  // }
  htot->Sumw2();
  for(int ibin=0; ibin<BinTotal; ibin++){
    double Nm = h_MC->GetBinContent(ibin+1);
    double NmErr = h_MC->GetBinError(ibin+1);
    double RelErr=Nm>0?NmErr*1.0/Nm:0.0;
    htot_Norm->SetBinContent(ibin+1, 1);
    //    htot_Norm->SetBinError(ibin+1, RelErr);
    double Nd = h_data->GetBinContent(ibin+1);
    double ScaledRelErr = Nm>0.?Nd/Nm*RelErr:0.0;
    htot_Norm->SetBinError(ibin+1, ScaledRelErr);
  }

  double maxY=max(h_data->GetMaximum(),h_MC->GetMaximum());
  double minY=min(h_data->GetMinimum(),h_MC->GetMinimum());
  //  h_data->GetYaxis()->SetRangeUser(0.95*minY, 1.2*maxY);
  // h_data->GetYaxis()->SetRangeUser(0.95*minY, 1.05*maxY);
  h_data->GetYaxis()->SetRangeUser(0., 1.5*maxY);
  //h_data->SetMaximum(1.2*maxY);
  if(IfLogY==1) h_data->GetYaxis()->SetRangeUser(0.1, 100*maxY);

  h_data->SetLineColor(1);
  h_data->SetFillColor(0);
  h_data->SetLineStyle(1);
  h_data->SetLineWidth(2);
  //h_data->GetXaxis()->SetTitle(XTitle.c_str());
  double WidthBin=(BinXHig-BinXLow)/BinTotal;
  //TString TitleY( Form("A.U. / %.2g GeV",WidthBin) );
  //TString TitleY( Form("No. of Entries in data / %.2g GeV",WidthBin) );
  //TString TitleY = "A.U";
  string PreTitleY( Form("Events / %.2g ",WidthBin) );
  //  string PreTitleY( Form("No. of Entries / %.2g ",WidthBin) );
  string TitleY =  PreTitleY + YUnit;
  h_data->GetYaxis()->SetTitle(TitleY.c_str());

  h_data->SetTitleSize(0.06,"X");
  h_data->SetTitleSize(0.06,"Y");
  //h_data->SetTitleOffset(1.3, "Y");
  h_data->SetTitleOffset(1.1, "Y");

  h_data->SetMarkerColor(kBlack);
  //h_data->SetMarkerSize(1.0);
  h_data->SetMarkerSize(0.8);
  h_data->SetMarkerStyle(20);

  h_MCsig->SetLineColor(2);
  h_MCsig->SetLineWidth(3);
  h_MC->SetBinContent(BinXHig, h_MC->GetBinContent(BinXHig)+h_MC->GetBinContent(BinXHig+1));
  // h_data->SetBinContent(BinXHig, h_data->GetBinContent(BinXHig)+h_data->GetBinContent(BinXHig+1));
  h_MC->SetFillColor(0);
  h_MC->SetMarkerStyle(0);
  h_MC->SetLineColor(1);
  h_MC->SetLineStyle(1);
  h_MC->SetLineWidth(2);
  TLegend *legend=new TLegend(0.17,0.7,0.89,0.9);
  legend->SetFillColor(0);
  legend->SetNColumns(3);
  legend->AddEntry(h_data,"Data","pe");


  legend->AddEntry(h_MCgg80toInf,"#gamma#gamma+jets","f");
  legend->AddEntry(h_MCsig,"M500signal*20000","l");

  legend->AddEntry(h_MCQCD40Inf,"datadriven 1&2 fake photons","f");
  legend -> SetTextSize(0.05);

  legend->SetTextFont(132); 

  // if(IfStatErr==1) 
  legend->AddEntry(htot_Norm, " MC Stat. Err.","f");

  //prepare 2 pads
  const Int_t nx=1;
  const Int_t ny=2;
  const Double_t x1[2] = {0.0,0.0};
  const Double_t x2[2] = {1.0,1.0};
  //const Double_t y1[] = {1.0,0.3};
  //const Double_t y2[] = {0.3,0.00};
  const Double_t y1[2] = {0.3,0.0};
  const Double_t y2[2] = {1.0,0.3};
  Double_t psize[2];
  TPad *pad;
  const char *myname = "c";
  char *name2 = new char [strlen(myname)+6];
  Int_t n = 0;
  for (int iy=0;iy<ny;iy++) {
    for (int ix=0;ix<nx;ix++) {
      n++;
      sprintf(name2,"%s_%d",myname,n);
      if(ix==0){
        gStyle->SetPadLeftMargin(.166);
      }else{
        gStyle->SetPadLeftMargin(.002);
        gStyle->SetPadTopMargin(.002);
      }

      if(iy==0){//upper
        gStyle->SetPadTopMargin(0.05*(1./0.7)); // 0.05
        gStyle->SetPadBottomMargin(.02);
      }
      if(iy==(ny-1)){//lower pad
        gStyle->SetPadTopMargin(.05);
        //gStyle->SetPadBottomMargin(.13*(1./0.3));
        gStyle->SetPadBottomMargin(.40);


      }
      pad = new TPad(name2,name2,x1[ix],y1[iy],x2[ix],y2[iy]);
      pad->SetNumber(n);
      pad->Draw();
      psize[iy]=y1[iy]-y2[iy];
      //if(iy>0 )pad->SetGrid(kTRUE);
    }// end of loop over x
  }// end of loop over y
  delete [] name2;

  //===Drawing====
  gPad->SetLeftMargin(0.18);
  gPad->SetBottomMargin(0.15);
  gPad->SetTopMargin(0.05);
  gPad->SetRightMargin(0.05);

  c1->SetFrameBorderSize(0);
  c1->SetFrameBorderMode(0);
  h_data->GetXaxis()->SetLabelColor(0);
  h_data->SetNdivisions(510 ,"X");

  c1->cd(1);
  gPad->SetLogy(IfLogY);
  //gPad->SetTickx(1);
  //gPad->SetTicky(1);
  //=========
  h_data->Draw("PE1");
  legend->Draw("same");
  hs->Draw("hist,same");
  h_MCsig->Draw("hist,same");
  h_signal->Draw("hist,same");

  //h_MC->Draw("hist,same");
  // if(IfStatErr==1){
  // }
  
  h_data->Draw("samePE1");
  h_data->Draw("Axissame");
  // check different bin content
  cout << "  1 bin:" << h_MC->GetBinContent(1) << endl;

  /*
    TLatex a;
    a.SetNDC();
    a.SetTextSize(0.05);
    a.DrawLatex(0.2,0.94, PrintInfor);
  */
  const TString PrintInfor1="#bf{CMS} #it{} #it{Preliminary}";
  const TString PrintInfor2="41.5 fb^{-1} (13TeV)"; 
  //tex = new TLatex(0.129,0.93, PrintInfor1);
  //TLatex *tex1 = new TLatex(0.16,0.94, PrintInfor1);
  TLatex *tex1 = new TLatex(0.25,0.94,PrintInfor1);
  tex1->SetNDC();
  tex1->SetTextFont(42);
  tex1->SetTextSize(0.045);
  tex1->SetLineWidth(2);
  tex1->Draw();

  TLatex *tex2 = new TLatex(0.56,0.933, PrintInfor2);
  tex2 = new TLatex(0.70,0.94, PrintInfor2);
  tex2->SetNDC();
  tex2->SetTextFont(42);
  tex2->SetTextSize(0.045);
  tex2->SetLineWidth(2);
  tex2->Draw();

  ///====
  TLine *Line1 = new TLine(h_data->GetBinLowEdge(1),1,h_data->GetBinLowEdge(h_data->GetNbinsX())+ h_data->GetBinWidth(h_data->GetNbinsX()),1);
  Line1->SetLineColor(1);
  Line1->SetLineWidth(2);
  Line1->SetLineStyle(4);

  TH1D *histoRatio = new TH1D(*h_data);
  histoRatio->Divide(h_data, h_MC, 1., 1.);
    // if(IfStatErr==1){
    //   }
    // }
  for(int ibin=0; ibin<BinTotal; ibin++){
    double Nd = h_data->GetBinContent(ibin+1);
    double NdErr = h_data->GetBinError(ibin+1);
    double Nm = h_MC->GetBinContent(ibin+1);
    histoRatio->SetBinError(ibin+1, Nd/Nm * NdErr/Nd);
    // histoRatio->SetBinError(ibin+1, Nd/Nm * NdErr/Nd);
  }
  for(int ibin=0; ibin<BinTotal; ibin++){
    double Nd = h_MC->GetBinContent(ibin+1);
    double NdErr = h_MC->GetBinError(ibin+1);
    h_MC->SetBinError(ibin+1, NdErr/Nd);
  }
  histoRatio->SetLineColor(1);
  histoRatio->SetLineStyle(1);
  histoRatio->SetLineWidth(2);
  histoRatio->SetMarkerColor(1);
  histoRatio->SetMarkerStyle(20);

  c1->cd(2);
  gPad->SetLogy(0);
  histoRatio->SetTitleOffset(1,"X");
  histoRatio->SetTitleSize(0.12,"X");
  histoRatio->SetLabelSize(0.1,"X");
  histoRatio->GetXaxis()->SetTitle(XTitle.c_str());
  histoRatio->GetYaxis()->SetTitle("Data / MC");
  //  histoRatio->GetYaxis()->SetTitle("data/MC");
  //histoRatio->SetTitleOffset(0.5,"Y");
  histoRatio->SetTitleOffset(0.4,"Y");
  //histoRatio->SetTitleSize(0.12,"Y");
  histoRatio->SetTitleSize(0.14,"Y");
  histoRatio->SetLabelSize(0.1,"Y");
  histoRatio->SetLabelColor(1,"X");
  histoRatio->GetYaxis()->CenterTitle();
  //histoRatio->SetNdivisions(505 ,"Y");
  histoRatio->SetNdivisions(510 ,"X");

  gPad->SetTickx(1);
  gPad->SetTicky(1);
 
  histoRatio->GetXaxis()->SetTickLength(0.08);
  histoRatio->GetYaxis()->SetTickLength(0.06);
  histoRatio->GetYaxis()->SetNdivisions(503);

  //histoRatio->SetMinimum(0.8); //0.5
  //histoRatio->SetMaximum(1.2);  //1.5
  histoRatio->SetMinimum(0.);
  histoRatio->SetMaximum(2.);
  histoRatio->Draw();
  // htot->SetFillColorAlpha(kRed,0.35);
  // htot->SetMarkerStyle(1);
  // htot->Draw("same:E2");
  htot_Norm->SetFillColorAlpha(kRed,0.35);
  htot_Norm->SetMarkerStyle(0);
  htot_Norm->Draw("same:E2");
    // if(IfStatErr==1){
    // }
  Line1->Draw("same");

  //===================================
  string nameplots=OutputPlotDir + "/DataMC_"+PlotName+".png";
  c1->Print(nameplots.c_str());

  // string nameplotspdf=OutputPlotDir + "/DataMC_"+PlotName+".pdf";
  // c1->Print(nameplotspdf.c_str());

  //c1->Clear();
  //legend->Clear();
 
}


void DataMCplot_cat7dd(){

  gROOT->ProcessLine(".x hggPaperStyle.C");
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(111);	
  // DrawMyPlots("Diphoton_mass", Preselections, "m_{#gamma#gamma} (GeV)", "GeV", "DiphotonMass", 18, 100., 180., 0, 0);
  // DrawMyPlots("LeadPhoton_pt", Preselections, "LeadPhoton_{pt} (GeV)", "GeV", "LeadPhoton_pt", 15, 30., 180., 0, 0);
  // DrawMyPlots("LeadPhoton_eta", Preselections, "LeadPhoton_{eta} (GeV)", "GeV", "LeadPhoton_eta", 15, -2.5, 2.5, 0, 0);
  // DrawMyPlots("LeadPhoton_phi", Preselections, "LeadPhoton_{phi} (GeV)", "GeV", "LeadPhoton_phi", 20, -4., 4., 0, 0);
  // DrawMyPlots("SubleadPhoton_pt", Preselections, "SubleadPhoton_{pt} (GeV)", "GeV", "SubleadPhoton_pt", 15, 20., 180., 0, 0);
  // DrawMyPlots("SubleadPhoton_eta", Preselections, "SubleadPhoton_{eta} (GeV)", "GeV", "SubleadPhoton_eta", 15, -2.5, 2.5, 0, 0);
  // DrawMyPlots("SubleadPhoton_phi", Preselections, "SubleadPhoton_{phi} (GeV)", "GeV", "SubleadPhoton_phi", 20, -4,4, 0, 0);
  // DrawMyPlots("jet_1_pt", Preselections, "LeadJet_{pt} (GeV)", "GeV", "jet_1_pt", 15, 20., 180., 0, 0);
  // DrawMyPlots("jet_1_E", Preselections, "LeadJet_{E} (GeV)", "GeV", "jet_1_E", 20, 0., 400., 0, 0);
  // DrawMyPlots("jet_2_E", Preselections, "SubLeadJet_{E} (GeV)", "GeV", "jet_2_E", 20, 0., 400., 0, 0);
  // DrawMyPlots("jet_3_E", Preselections, "Jet3_{E} (GeV)", "GeV", "jet_3_E", 20, 0., 400., 0, 0);
  // DrawMyPlots("jet_4_E", Preselections, "Jet4_{E} (GeV)", "GeV", "jet_2_E", 20, 0., 400., 0, 0);
  // DrawMyPlots("jet_2_pt", Preselections, "SubLeadJet_{pt} (GeV)", "GeV", "jet_2_pt", 20, 0., 400., 0, 0);
  // DrawMyPlots("jet_3_pt", Preselections, "Jet3_{pt} (GeV)", "GeV", "jet_3_pt", 20, 0., 400., 0, 0);
  // DrawMyPlots("jet_4_pt", Preselections, "Jet4_{pt} (GeV)", "GeV", "jet_2_pt", 20, 0., 400., 0, 0);
  // DrawMyPlots("nGoodAK4jets", Preselections, "nGoodAK4jets", " ", "nGoodAK4jets", 9, 0., 9., 0, 0);

  // DrawMyPlots("PuppiMET_pt", Preselections, "PuppiMET", "GeV", "PuppiMET", 20, 0., 150., 0, 0);
  // DrawMyPlots("LeadPhoton_eta", Preselections, "LeadPhoton_eta", "", "LeadPhoton_eta", 15, -2.5, 2.5, 0, 0);
  // DrawMyPlots("SubleadPhoton_eta", Preselections, "SubleadPhoton_eta", "", "SubleadPhoton_eta", 15, -2.5, 2.5, 0, 0);
  // DrawMyPlots("Diphoton_minID", Preselections, "Diphoton_minID", "", "Diphoton_minID", 17, -0.7, 1, 0, 0);
  // DrawMyPlots("Diphoton_maxID", Preselections, "Diphoton_maxID", "", "Diphoton_maxID", 15, -0.7, 1, 0, 0);
  // DrawMyPlots("Diphoton_phi", Preselections, "Diphoton_phi", "", "Diphoton_phi", 15,-4., 4., 0, 0);
  // DrawMyPlots("Diphoton_dR", Preselections, "Diphoton_dR", "", "Diphoton_dR", 15, 0., 4., 0, 0);
  // DrawMyPlots("mindR_gg_4jets", Preselections, "mindR_gg_4jets", "", "mindR_gg_4jets", 15, 0., 4., 0, 0);
  // DrawMyPlots("maxdR_gg_4jets", Preselections, "maxdR_gg_4jets", "", "maxdR_gg_4jets", 15, 2., 6., 0, 0);
  // DrawMyPlots("maxdR_4jets", Preselections, "maxdR_4jets", "", "maxdR_4jets", 15, 0., 6., 0, 0);
  // DrawMyPlots("mindR_4jets", Preselections, "mindR_4jets", "", "mindR_4jets", 15, 0., 4., 0, 0);
  // DrawMyPlots("costhetastar", Preselections, "costhetastar", "", "costhetastar", 12, 0., 1., 0, 0);
  // DrawMyPlots("costheta1", Preselections, "costheta1", "", "costheta1", 20, 0., 1., 0, 0);
  // DrawMyPlots("costheta2", Preselections, "costheta2", "", "costheta2", 20, 0., 1., 0, 0);
  // DrawMyPlots("scaled_subleadphoton_pt", Preselections, "scaled_subleadphoton_pt (GeV)", "GeV", "scaled_subleadphoton_pt", 15, 0., 1., 0, 0);
  // DrawMyPlots("scaled_leadphoton_pt", Preselections, "scaled_leadphoton_pt (GeV)", "GeV", "scaled_leadphoton_pt", 15, 0.2, 1.5, 0, 0);
  // DrawMyPlots("W1_E", Preselections, "W1_{E} (GeV)", "GeV", "W1_E", 27, 30., 300., 0, 0);
  // DrawMyPlots("W1_pt", Preselections, "W1_{pt} (GeV)", "GeV", "W1_pt", 15, 0., 200., 0, 0);
  // DrawMyPlots("W1_mass", Preselections, "W1_{mass} (GeV)", "GeV", "W1_mass", 15, 0., 180., 0, 0);
  // DrawMyPlots("W1_eta", Preselections, "W1_{eta} (GeV)", "GeV", "W1_eta",  15, -2.5, 2.5, 0, 0);
  // DrawMyPlots("W1_phi", Preselections, "W1_{phi} (GeV)", "GeV", "W1_phi", 15,-4., 4., 0, 0);
  // DrawMyPlots("W2_E", Preselections, "W2_{E} (GeV)", "GeV", "W2_E", 17, 30., 200., 0, 0);
  // DrawMyPlots("W2_pt", Preselections, "W2_{pt} (GeV)", "GeV", "W2_pt", 15, 0., 180., 0, 0);
  // DrawMyPlots("W2_mass", Preselections, "W2_{mass} (GeV)", "GeV", "W2_mass", 15, 0., 180., 0, 0);
  // DrawMyPlots("W2_eta", Preselections, "W2_{eta} (GeV)", "GeV", "W2_eta",  15, -2.5, 2.5, 0, 0);
  // DrawMyPlots("W2_phi", Preselections, "W2_{phi} (GeV)", "GeV", "W2_phi", 15,-4., 4., 0, 0);
  // DrawMyPlots("WW_E", Preselections, "WW_{E} (GeV)", "GeV", "WW_E", 15, 100., 400., 0, 0);
  // DrawMyPlots("WW_pt", Preselections, "WW_{pt} (GeV)", "GeV", "WW_pt", 15, 0., 180., 0, 0);
  // DrawMyPlots("WW_mass", Preselections, "WW_{mass} (GeV)", "GeV", "WW_mass", 20,50., 450., 0, 0);
  // DrawMyPlots("WW_eta", Preselections, "WW_{eta} (GeV)", "GeV", "WW_eta",  15, -2.5, 2.5, 0, 0);
  // DrawMyPlots("WW_phi", Preselections, "WW_{phi} (GeV)", "GeV", "WW_phi", 15,-4., 4., 0, 0);
  DrawMyPlots("dphi_j1j2", Preselections, "dphi_j1j2", " ", "j1j2_phi", 15,0., 7., 0, 0);
  DrawMyPlots("dphi_j1j3", Preselections, "dphi_j1j3", " ", "j1j3_phi", 15,0., 7., 0, 0);
  DrawMyPlots("dphi_j1j4", Preselections, "dphi_j1j4", " ", "j1j4_phi", 15,0., 7., 0, 0);
  DrawMyPlots("dphi_j2j3", Preselections, "dphi_j2j3", " ", "j2j3_phi", 15,0., 7., 0, 0);
  DrawMyPlots("dphi_j2j4", Preselections, "dphi_j2j4", " ", "j2j4_phi", 15,0., 7., 0, 0);
  DrawMyPlots("dphi_j3j4", Preselections, "dphi_j3j4", " ", "j3j4_phi", 15,0., 7., 0, 0);
  
  // DrawMyPlots("WW_phi", Preselections, "WW_{phi} (GeV)", "GeV", "WW_phi", 15,-4., 4., 0, 0);
  // DrawMyPlots("sum_two_max_bscore", Preselections, "sum_two_max_bscore", " ", "sum_two_max_bscore", 15, 0., 1.5, 0, 0);
  // DrawMyPlots("dnn_score",  Preselections, "DNN score", "", "DNN_score", 20, 0.8, 1., 0, 0);


  return;
 

} 

  // DrawMyPlots("fatjet_H_Hqqqq_qqlv_vsQCDTop", Preselections, "PN tagger (GeV)", "GeV", "PN_tagger", 15, 0.2, 1., 0, 0);
  // DrawMyPlots("dnn_score",  Preselections, "DNN score", "", "DNN_score", 30, 0.1, 1., 0, 1);
