import matplotlib.pyplot as plt
import json
import pandas as pd
import gc
import sys
import os
import numpy as np
# plotdir=sys.argv[1]
plotdir="/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7"

# useXGB=sys.argv[2]
useXGB=bool(False)
# exts=["1_features","2_features","3_features","4_features","5_features","6_features","7_features","8_features","9_features","10_features","11_features","12_features","13_features","14_features","15_features","16_features","17_features","18_features","19_features","20_features"]
exts=["55_features"]
if(useXGB=="True"):
    useXGB=True
else:
    useXGB=False
os.system("mkdir -p %s"%plotdir)
def clearVar(var):
    del var
    gc.collect()
mass_list=["pointsName"]
# mass_list=[100,125,150,170,190,250,300,350,400,450,500,550,600,60,700,70,80,90]
# mass_list=[70]

print(useXGB)


def getFprTpr(predictions,trueValues):
    TPR=[]
    FPR=[]
    for threshold in np.arange(0,1.1,0.1):
        # print(threshold)
        BinaryList = [1 if i > threshold else 0 for i in predictions]
        PlusResult = [x + y for x, y in zip(BinaryList, trueValues)]
        MinusResult = [x - y for x, y in zip(BinaryList, trueValues)]
        TP=PlusResult.count(2)
        TN=PlusResult.count(0)
        FP=MinusResult.count(1)
        FN=MinusResult.count(-1)
        TPR.append(TP/(TP+FN))
        FPR.append(FP/(FP+TN))
        print("The threshold is %s, TPR is %s, FPR is %s"%(threshold,TP/(TP+FN),FP/(FP+TN)))
    return TPR,FPR






for ext in exts:
    for points in mass_list:
        print("Now:",points)
        if (useXGB):
            points="Relu_BDT_"+str(points)
        else:
            points="Relu_"+ext+"_50epochs_"+str(points)
        if(not useXGB ):
            history = open("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/history_%s.json"%points, 'r', encoding='utf-8')
            history=json.load(history)
            loss = history["loss"]
            val_loss = history["val_loss"]

            # # 计算损失曲线的值
            epochs = range(1, len(loss) + 1)

            # # 画出损失曲线
            plt.plot(epochs, loss, "bo", label="Training loss")
            plt.plot(epochs, val_loss, "b", label="Validation loss")
            plt.title("Training and validation loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.legend()
            plt.savefig("%s/loss_%s_log.png"%(plotdir,points))
            plt.clf()
            clearVar(history)

        predict = open("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/predictions_%s.json"%points, 'r', encoding='utf-8')
        prediction=json.load(predict)

        plt.plot(prediction["test_fpr"],prediction["test_tpr"],label="test ROC(AUC= %0.3f)"%prediction["auc_test"])
        plt.plot(prediction["train_fpr"],prediction["train_tpr"],label="train ROC(AUC= %0.3f)"%prediction["auc"])
        plt.legend()
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.title(ext)
        plt.savefig("%s/ROC_%s.png"%(plotdir,points))
        plt.clf()
        clearVar(predict)
        clearVar(prediction)



        df_test=pd.read_csv("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/df_train_%s.csv"%points)
        print("Train weighted",df_test.query("train_tag==0").weights.sum(),df_test.query("train_tag==1").weights.sum())
        print("Train unweighted",df_test.query("train_tag==0").shape[0],df_test.query("train_tag==1").shape[0])
        
        plt.hist(df_test.query("train_tag==0").predictions,log=True,label="bkg",bins=np.linspace(0,1,11))
        plt.hist(df_test.query("train_tag==1").predictions,log=True,label="signal",bins=np.linspace(0,1,11),alpha=0.7)
        plt.legend()
        
        plt.savefig("%s/Train_distributions_%s.png"%(plotdir,points))
        plt.clf()
        plt.hist(df_test.query("train_tag==0").weights,log=True,label="bkg")
        plt.hist(df_test.query("train_tag==1").weights,log=True,label="signal",alpha=0.7)
        plt.legend()
        plt.savefig("%s/Weight_distributions_%s.png"%(plotdir,points))
        plt.clf()
        TPR,FPR=getFprTpr(df_test.predictions,df_test.train_tag)
        

        df_test=pd.read_csv("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/df_test_%s.csv"%points)
        print("test weighted",df_test.query("test_tag==0").weights.sum(),df_test.query("test_tag==1").weights.sum())
        print("test unweighted",df_test.query("test_tag==0").shape[0],df_test.query("test_tag==1").shape[0])
        plt.hist(df_test.query("test_tag==0").predictions_test,log=True,label="bkg",bins=np.linspace(0,1,11))
        plt.hist(df_test.query("test_tag==1").predictions_test,log=True,label="signal",bins=np.linspace(0,1,11),alpha=0.7)
        plt.legend()
        plt.savefig("%s/Test_distributions_%s.png"%(plotdir,points))
        plt.clf()

        TPR_test,FPR_test=getFprTpr(df_test.predictions_test,df_test.test_tag)

        plt.plot(FPR_test,TPR_test,"bo",label="test ROC")
        plt.plot(FPR,TPR,label="train ROC")
        plt.legend()
        plt.xlabel("fpr")
        plt.ylabel("tpr")

        plt.savefig("%s/ROC_cal_%s.png"%(plotdir,points))
        plt.clf()
