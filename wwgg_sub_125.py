
import tensorflow as tf
import gc
import platform
import sys
import json
import xgboost as xgb
from functools import partial
from xgboost import plot_tree
# from sklearn.utils import compute_class_weight
# from plotting.plotter import plotter
from sklearn import metrics
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt
# print(f"Python Platform: {platform.platform()}")
# print(f"Tensor Flow Version: {tf.__version__}")
# print(f"Keras Version: {tf.keras.__version__}")

# gpu = len(tf.config.list_physical_devices('GPU'))>0
# print("GPU is", "available" if gpu else "NOT AVAILABLE")
# print("chuw:",(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import numpy as np

import glob
import pandas as pd
import matplotlib.pyplot as plt
import sys

def load_trained_model(model_path):
    print('<load_trained_model> weights_path: ', model_path)
    model = load_model(model_path, compile=False)
    return model


def clearVar(var):
    del var
    gc.collect()


bkgFiles=glob.glob("/hpcfs/cms/cmsgpu/zhangjie/DNN/input/*parquet")
sigFiles=glob.glob("/hpcfs/cms/cmsgpu/zhangjie/DNN/input/signal_cut/*parquet")




AllSample="125"
def readDF(f,processID,isBkg:bool):
    df=pd.read_parquet(f)
    df=df.query("minID>-0.7")
    if(isBkg):
        df["target"]=0
        df["weight"]=df["weight"]
    else:
        df["target"]=1
    df["processID"]=processID
    # print("file name is:",f.split("/")[-1],"number of events:",df.shape[0],"weighted events:",df.weight.sum())
    return df

def getClassWeight(df):
    bkg_yields=df.groupby("target")["weight"].sum()[0]
    sig_yields=df.groupby("target")["weight"].sum()[1]
    N_bkg=df.query("target==0").shape[0]
    N=bkg_yields
    # N=N_bkg
    norm_bkg=N/bkg_yields
    norm_sig=N/sig_yields
    # norm=sig_yields/bkg_yields
    df["class_weight"]=df["target"].map({0:norm_bkg, 1:norm_sig})
    # df["class_weight"]=df["target"].map({0:norm, 1:1})
    df["new_weight"]= df["class_weight"]* df["weight"]
    # df["new_weight"]= 1 

    print("original wighted events:",bkg_yields,sig_yields)
    print("new weight:",df.groupby("target")["new_weight"].sum())
    return df,norm_bkg,norm_sig





import re
def getMXMY(filename):
    mx_my = re.search("MX-(\d+)_MY-(\d+)", filename)

    if mx_my:
        mx = mx_my.group(1)
        my = mx_my.group(2)
    return "Signal_"+str(mx)+"_"+str(my)+"_point"




signals=[]
for f in sigFiles:
    processID=getMXMY(f)
    print(processID)
    df=readDF(f,processID,False)
    signals.append(df)
signal=pd.concat(signals)
# if (AllSample!="all"):
    # signal=signal.query("Ymass == 125")



QCD=readDF('/hpcfs/cms/cmsgpu/zhangjie/DNN/input/datadriven_rename.parquet',"BKG_QCD",True)


Dipho=readDF('/hpcfs/cms/cmsgpu/zhangjie/DNN/input/diphoton.parquet',"BKG_diphoton",True)

data_set=pd.concat([QCD,Dipho,signal])
data_set=data_set.fillna(-999)
data_set=data_set.query("res_cosTheta_CS>-1")
# clearVar(QCD)
# clearVar(Dipho)
# clearVar(signal)






# training_columns=["new_weight","Ymass","Diphoton_DiphoCosThetaStar","cosdijet","res_cosTheta_CS","dijet_pt_mggjj","Diphoton_dipho_pt_mggjj_res","Diphoton_dR","Diphoton_lead_pt_mgg","Diphoton_sublead_pt_mgg",
#                   "dijet_sigmoM","leadjet_pt_mjj","subleadjet_pt_mjj","leadjet_btagDeepFlavB","subleadjet_btagDeepFlavB","Diphoton_lead_pho_sigEoE","Diphoton_sublead_pho_sigEoE","res_mindr",
#                   "otherminjetphoton","Leading_Photon_MVA","Subleading_Photon_MVA"
#                  ]
allFeatures=["res_cosTheta_CS"
                 ]

# training_columns=["Diphoton_DiphoCosThetaStar","cosdijet","res_cosTheta_CS","dijet_pt_mggjj","Diphoton_dipho_pt_mggjj_res",
                #  ] #0.9160
# training_columns=["Diphoton_DiphoCosThetaStar","cosdijet","res_cosTheta_CS","dijet_pt_mggjj","Diphoton_dipho_pt_mggjj_res","Leading_Photon_MVA","Subleading_Photon_MVA"
#                  ]#0.9059
# training_columns=["Diphoton_DiphoCosThetaStar","cosdijet","res_cosTheta_CS","dijet_pt_mggjj","Diphoton_dipho_pt_mggjj_res","Leading_Photon_MVA","Subleading_Photon_MVA","leadjet_btagDeepFlavB","subleadjet_btagDeepFlavB"
#                  ]#0.9943
# training_columns=["Diphoton_DiphoCosThetaStar","cosdijet","res_cosTheta_CS","dijet_pt_mggjj","Diphoton_dipho_pt_mggjj_res","Leading_Photon_MVA","Subleading_Photon_MVA","leadjet_btagDeepFlavB","subleadjet_btagDeepFlavB","leadjet_pt_mjj","subleadjet_pt_mjj"
#                  ]#0.9941
# training_columns=["new_weight","Ymass","Diphoton_DiphoCosThetaStar","cosdijet","res_cosTheta_CS","dijet_pt_mggjj","Diphoton_dipho_pt_mggjj_res","Leading_Photon_MVA","Subleading_Photon_MVA","leadjet_btagDeepFlavB","subleadjet_btagDeepFlavB","leadjet_pt_mjj","subleadjet_pt_mjj","Diphoton_lead_pt_mgg","Diphoton_sublead_pt_mgg"
                #  ]#0.9973
# training_columns=["Diphoton_DiphoCosThetaStar","cosdijet","res_cosTheta_CS","dijet_pt_mggjj","Diphoton_dipho_pt_mggjj_res","Leading_Photon_MVA","Subleading_Photon_MVA","leadjet_btagDeepFlavB","subleadjet_btagDeepFlavB","leadjet_pt_mjj","subleadjet_pt_mjj","Diphoton_lead_pt_mgg","Diphoton_sublead_pt_mgg",
#                     "dijet_sigmoM"
#                  ]#0.9969
# training_columns=["Diphoton_DiphoCosThetaStar","cosdijet","res_cosTheta_CS","dijet_pt_mggjj","Diphoton_dipho_pt_mggjj_res","Leading_Photon_MVA","Subleading_Photon_MVA","leadjet_btagDeepFlavB","subleadjet_btagDeepFlavB","leadjet_pt_mjj","subleadjet_pt_mjj","Diphoton_lead_pt_mgg","Diphoton_sublead_pt_mgg",
#                     "dijet_sigmoM","Diphoton_lead_pho_sigEoE","Diphoton_sublead_pho_sigEoE"
#                  ]#0.9966
# training_columns=["Diphoton_DiphoCosThetaStar","cosdijet","res_cosTheta_CS","dijet_pt_mggjj","Diphoton_dipho_pt_mggjj_res","Leading_Photon_MVA","Subleading_Photon_MVA","leadjet_btagDeepFlavB","subleadjet_btagDeepFlavB","leadjet_pt_mjj","subleadjet_pt_mjj","Diphoton_lead_pt_mgg","Diphoton_sublead_pt_mgg",
#                     "dijet_sigmoM","Diphoton_lead_pho_sigEoE","Diphoton_sublead_pho_sigEoE","res_mindr","otherminjetphoton"
#                  ]#0.9972
# training_columns=["Diphoton_DiphoCosThetaStar","cosdijet","res_cosTheta_CS","dijet_pt_mggjj","Diphoton_dipho_pt_mggjj_res","Leading_Photon_MVA","Subleading_Photon_MVA","leadjet_btagDeepFlavB","subleadjet_btagDeepFlavB","leadjet_pt_mjj","subleadjet_pt_mjj","Diphoton_lead_pt_mgg","Diphoton_sublead_pt_mgg",
#                     "dijet_sigmoM","Diphoton_lead_pho_sigEoE","Diphoton_sublead_pho_sigEoE","res_mindr","otherminjetphoton","Diphoton_dR"
#                  ]#0.9972

training_columns=["new_weight","Ymass","Diphoton_DiphoCosThetaStar","cosdijet","Diphoton_lead_pt_mgg","Diphoton_sublead_pt_mgg","leadjet_pt_mjj","subleadjet_pt_mjj","Leading_Photon_MVA","Subleading_Photon_MVA","Diphoton_dipho_pt_mggjj_res","dijet_pt_mggjj",
                  "dijet_sigmoM","Diphoton_lead_pho_sigEoE","Diphoton_sublead_pho_sigEoE","res_mindr",
                  "otherminjetphoton","Diphoton_dR"
                 ]#0.9973
# training_columns=["new_weight","Ymass","bjet_1_btagDeepFlavB","bjet_2_btagDeepFlavB","Diphoton_DiphoCosThetaStar","cosdijet","Diphoton_lead_pt_mgg","Diphoton_sublead_pt_mgg","leadjet_pt_mjj","subleadjet_pt_mjj","Leading_Photon_MVA","Subleading_Photon_MVA","Diphoton_dipho_pt_mggjj_res","dijet_pt_mggjj",
#                   "dijet_sigmoM","Diphoton_lead_pho_sigEoE","Diphoton_sublead_pho_sigEoE","res_mindr",
#                   "otherminjetphoton","Diphoton_dR"
#                  ]#0.9973



# training_columns=["Leading_Photon_MVA","Subleading_Photon_MVA","leadjet_btagDeepFlavB","subleadjet_btagDeepFlavB"]
def baseline_model(num_variables,learn_rate=0.000000001):
    model = Sequential()
    # model.add(tf.keras.layers.Dense(64,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    model.add(tf.keras.layers.Dense(32,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    #model.add(Dense(64,activation='relu'))
    # model.add(tf.keras.layers.Dense(32,activation='relu'))
    model.add(tf.keras.layers.Dense(16,activation='relu'))
    model.add(tf.keras.layers.Dense(8,activation='relu'))
    model.add(tf.keras.layers.Dense(4,activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=learn_rate),metrics=['acc'])
    # optimizer=tf.optimizers.SGD(learn_rate)
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model
def leaky_model(num_variables,learn_rate=0.000000001):

    # model.add(tf.keras.layers.Dense(64,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    # model.add(tf.keras.layers.Dense(32,activation='relu'))

    model = Sequential()
    model.add(tf.keras.layers.Dense(10, input_dim=num_variables,kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.layers.LeakyReLU(alpha=0.001)))

    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.layers.LeakyReLU(alpha=0.001)))
    model.add(tf.keras.layers.Dense(4))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.layers.LeakyReLU(alpha=0.001)))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid")) 
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def gscv_model(learn_rate=0.001):
    model = Sequential()
    model.add(tf.keras.layers.Dense(32,input_dim=len(training_columns)-1,kernel_initializer='glorot_normal',activation='relu'))
    model.add(tf.keras.layers.Dense(16,activation='relu'))
    model.add(tf.keras.layers.Dense(8,activation='relu'))
    model.add(tf.keras.layers.Dense(4,activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=learn_rate),metrics=['acc'])
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def new_model(num_variables,learn_rate=0.001):
    model = Sequential()
    model.add(tf.keras.layers.Dense(10, input_dim=num_variables,kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    #model.add(Dense(16,kernel_regularizer=regularizers.l2(0.01)))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(4))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid")) 
    optimizer=Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model



if (AllSample!="all"):
    data_set=data_set.query("Ymass==125")
data_set,norm_bkg,norm_sig=getClassWeight(data_set)
print("norm factor:",norm_bkg,norm_sig)



train_model=True
epochs=200
gridSearch=False
useXGB=False

def scheduler(epoch):
  if epoch < 10:
    return 0.0001
  else:
    return 0.0000001

# Create a LearningRateScheduler callback that uses the scheduler function
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)



for feature in allFeatures:
    training_columns.append(feature)
    ext="%s_features"%str(len(training_columns)-1)
    print("total %s features"%str(len(training_columns)-1))
    print("all features:",training_columns)
    X=data_set[training_columns].values
    y=data_set["target"].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,)
    print(len(y_train))
    print("Xtrain:",X_train)

    trainingweights=X_train[:,0]
    testweights=X_test[:,0]
    print("train weight:",trainingweights)
    print(len(trainingweights))

    X_train=X_train[:,1:]
    X_test=X_test[:,1:]

    print(X_train)

    if (useXGB):
        name="BDT_125"
    else:
        # name="lerky_"+ext+"_%sepochs_125"%str(epochs)
        name="Relu_"+ext+"_%sepochs_125"%str(epochs)
    print("Model extension name: %s"%name)
    if (train_model):
        print("start training")
        if (useXGB):
            print("Train BDT")
            dtrain = xgb.DMatrix(X_train, label=y_train,weight=trainingweights)
            params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'max_depth': 4,
        'lambda': 1,
        'subsample': 1,
        'colsample_bytree': 0.5,
        'eta': 0.1,
        'seed': 0,
        'silent': 1
            }
            model = xgb.train(params, dtrain, num_boost_round=100)
            model.save_model("/hpcfs/cms/cmsgpu/zhangjie/DNN/model_%s.xgb"%(name))
            gv = xgb.to_graphviz(model)
            gv.render("xgb-structure_%s"%(name), format="pdf")
            gv.render("xgb-structure_%s"%(name), format="png")
        else:
            if(gridSearch):
                learn_rates=[0.000001, 0.00001,0.0001,0.001,0.01]
                epochs = [150,200]
                batch_sizes = [50,100,200,400,500]
                param_grid = dict(learn_rate=learn_rates,epochs=epochs,batch_size=batch_sizes)
                print(param_grid)
                model = KerasClassifier(build_fn=gscv_model,verbose=0)
                grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
                grid_result = grid.fit(X_train,y_train,shuffle=True,sample_weight=trainingweights)
                print("Best score: %f , best params: %s" % (grid_result.best_score_,grid_result.best_params_))
            else:
                early_stopping_monitor = EarlyStopping(patience=100, monitor='val_loss', min_delta=0.001, verbose=1)
                # model = leaky_model(len(training_columns)-1)
                model = new_model(len(training_columns)-1)
                # history=model.fit(X_train,y_train,validation_split=0.2,epochs=epochs,batch_size=200,verbose=2,shuffle=False,sample_weight=trainingweights,class_weight={0:norm_bkg,1:norm_sig},callbacks=[early_stopping_monitor])
                history=model.fit(X_train,y_train,validation_split=0.2,epochs=epochs,batch_size=200,verbose=2,shuffle=True,sample_weight=trainingweights,callbacks=[early_stopping_monitor,lr_callback])
                # history=model.fit(X_train,y_train,validation_split=0.2,epochs=epochs,batch_size=200,verbose=2,shuffle=True,callbacks=[early_stopping_monitor])
                model.save("/hpcfs/cms/cmsgpu/zhangjie/DNN/model_%s.h5"%(name))
                print(type(history.history["loss"]))
                history_dict = {"loss":history.history["loss"],"val_loss":history.history["val_loss"]}

                print(history_dict)
                print(type(history_dict))
                with open("history_%s.json"%name, "w") as f:
                    json.dump(history_dict, f)

    else:
        print("Start predict")
        model=load_trained_model("/hpcfs/cms/cmsgpu/zhangjie/DNN/model_%s.h5"%name)



    print("Start predict_125")
    if (useXGB):
        predictions=model.predict(xgb.DMatrix(X_train))
        predictions_test=model.predict(xgb.DMatrix(X_test)) 
    else:
        predictions=model.predict(X_train)
        predictions_test=model.predict(X_test)
    # print(type(predictions_test))
    # print(predictions)
    # print(len(predictions.tolist()))
    fpr, tpr, thresholds = metrics.roc_curve(y_train, predictions)
    # print(len(fpr.tolist()))
    # print(len(y_train.tolist()))
    fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, predictions_test)
    # print("print true false:",fpr,tpr)
    # print(type(fpr_test))
    if(useXGB):
        predictions=predictions.tolist()
        predictions_test=predictions_test.tolist()    
    else:
        predictions=predictions[:,0].tolist()
        predictions_test=predictions_test[:,0].tolist()
    data_train = {
        'predictions': predictions,
        'train_tag':y_train.tolist(),
        'weights':trainingweights.tolist(),
        'Ymass':X_train[:,0].tolist()
    }
    data_test = {
        'predictions_test': predictions_test,
        'test_tag':y_test.tolist(),
        'weights':testweights.tolist(),
        'Ymass':X_test[:,0].tolist()
    }
    roc_train = {
        'train_fpr':fpr.tolist(),
        'train_tpr':tpr.tolist(),
        'thresholds':thresholds.tolist(),
        'thresholds_test':thresholds_test.tolist(),
        'test_fpr':fpr_test.tolist(),
        'test_tpr':tpr_test.tolist(),

    }

    # print(dict_predictions["predictions"])
    with open("predictions_%s.json"%name, "w") as f:
        json.dump(roc_train, f)
    df_train=pd.DataFrame(data_train)
    df_train.to_csv("df_train_%s.csv"%name)
    df_test=pd.DataFrame(data_test)
    df_test.to_csv("df_test_%s.csv"%name)


