
import tensorflow as tf
import gc
import platform
import sys
import json
import xgboost as xgb
from functools import partial
from xgboost import plot_tree
from sklearn import metrics
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
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
import random
import glob
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm.notebook import tqdm 
from IPython.display import display
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler
import pickle
import re
from scipy import integrate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
def load_trained_model(model_path):
    print('<load_trained_model> weights_path: ', model_path)
    model = load_model(model_path, compile=False)
    return model
def clearVar(var):
    del var
    gc.collect()
def readDF(f,processID,isBkg:bool):
    df=pd.read_parquet(f)

    # df=df.query("jet_1_btagDeepFlavB<=0.3093&jet_2_btagDeepFlavB<=0.3093")
   
    if(isBkg):
        # mass = [250, 260, 270, 280, 300, 320, 350, 400, 450,550, 600, 650, 700, 750, 800, 850, 900, 1000]
        mass = [250, 300, 400, 450,550, 650, 750, 900]
        random_mass = random.choices(mass, k=len(df))
        df["target"]=0
        df["weight"]=df["weight_central"]
        df['Signal_Mass']=random_mass
    else:
        mx = re.search("m(\d+)", f)
        df["target"]=1
        df['Signal_Mass']=int(mx.group(1))
        df["weight"]=df["weight_central"]


    df["processID"]=processID
    # print("file name is:",f.split("/")[-1],"number of events:",df.shape[0],"weighted events:",df.weight.sum())
    return df
def getClassWeight(df):
    #normalize signal to backgroud
    # background target == 0; sig target ==1
    bkg_yields=df.groupby("target")["weight"].sum()[0]
    sig_yields=df.groupby("target")["weight"].sum()[1]
    N_bkg=df.query("target==0").shape[0]
    N=bkg_yields
    norm_bkg=N/bkg_yields
    norm_sig=N/sig_yields
    # norm=sig_yields/bkg_yields
    df["class_weight"]=df["target"].map({0:norm_bkg, 1:norm_sig})
    df["new_weight"]= df["class_weight"]* df["weight"]

    print("original wighted events:",bkg_yields,sig_yields)
    print("new weight:",df.groupby("target")["new_weight"].sum())
    return df,norm_bkg,norm_sig
def getMX(filename):
    mx = re.search("m(\d+)", filename)
    if mx:
        mx = mx.group(1)
    return "Signal_M"+str(mx)+"_point"
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
    model.add(tf.keras.layers.Dense(80, input_dim=num_variables,kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    #添加批量归一化层，加速训练提高准确性
    model.add(tf.keras.layers.BatchNormalization())
    #非线性
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(80))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(80))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid")) 
    #梯度下降算法
    optimizer=Adam(lr=learn_rate)
    #损失函数：二元交叉熵，使用准确率作为评估指标
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model
# define learning rate 
def scheduler(epoch):
  if epoch < 10:
    return 0.0001
  else:
    return 0.0000001

allFeatures = ["Diphoton_dR"]

training_columns=["new_weight","scaled_leadphoton_pt","scaled_subleadphoton_pt","WW_pt","W1_mass","WW_mass","maxdR_gg_4jets","mindR_4jets","W2_mass","jet_1_pt","jet_2_pt","jet_4_pt","jet_4_E","jet_3_E","jet_2_E","jet_1_E","jet_3_pt","sum_two_max_bscore","LeadPhoton_eta","SubleadPhoton_eta","W1_pt","W2_pt","jet_1_eta","jet_2_eta","jet_3_eta","jet_4_eta","costhetastar","LeadPhoton_sigEoverE","SubleadPhoton_sigEoverE","Diphoton_pt","Diphoton_minID","Diphoton_maxID","nGoodAK4jets","costheta1","costheta2","Signal_Mass"]

data_set=pd.read_csv('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/dnn.csv',index_col=0)


train_model=True
epochs=30
gridSearch=False
useXGB=False
compute_importance=True

# Create a LearningRateScheduler callback that uses the scheduler function
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


for feature in allFeatures:
    training_columns.append(feature)
    ext="%s_features"%str(len(training_columns)-1)
    print("total %s features"%str(int(len(training_columns)-1)))
    print("all features:",training_columns)
    #define training var as X and target sig or background by y
    X=data_set[training_columns].values
    y=data_set["target"].values

# split train datset 70% and test dataset 30% from whole dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,)
    print(len(y_train))
    print("Xtrain:",X_train)

    trainingweights=X_train[:,0]
    testweights=X_test[:,0]
    print("train weight:",trainingweights)
    print(len(trainingweights))

    X_train=X_train[:,1:]
    X_test=X_test[:,1:]


#     ###### 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    traincsv = pd.DataFrame(X_train_scaled)
    testcsv=pd.DataFrame(X_test_scaled)

    traincsv.to_csv('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/train_scaled.csv')
    testcsv.to_csv('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/test_scaled.csv')
    # traincsv=pd.read_csv('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/train_scaled.csv',index_col=0)
    # testcsv=pd.read_csv('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/test_scaled.csv',index_col=0)
    with open('/hpcfs/cms/cmsgpu/shaoweisong/input/cat7/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)

    if (useXGB):
        name="BDT_pointsName"
    else:
        name="Relu_"+ext+"_%sepochs_pointsName"%str(epochs)
    print("Model extension name: %s"%name)


    if (train_model):
        print("start training")
        if (useXGB):
            print("Train BDT")
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train,weight=trainingweights)
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
            model.save_model("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/model_%s.xgb"%(name))
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
                grid_result = grid.fit(X_train_scaled,y_train,shuffle=True,sample_weight=trainingweights)
                print("Best score: %f , best params: %s" % (grid_result.best_score_,grid_result.best_params_))
            else:
                early_stopping_monitor = EarlyStopping(patience=100, monitor='val_loss', min_delta=0.001, verbose=1)
                #min_deltaR 降低
                # model = leaky_model(len(training_columns)-1)
                model = new_model(len(training_columns)-1)
                # history=model.fit(X_train_scaled,y_train,validation_split=0.2,epochs=epochs,batch_size=200,verbose=2,shuffle=False,sample_weight=trainingweights,class_weight={0:norm_bkg,1:norm_sig},callbacks=[early_stopping_monitor])
                history=model.fit(X_train_scaled,y_train,validation_split=0.2,epochs=epochs,batch_size=500,verbose=2,shuffle=True,sample_weight=trainingweights,callbacks=[early_stopping_monitor,lr_callback])
                # history=model.fit(X_train_scaled,y_train,validation_split=0.2,epochs=epochs,batch_size=200,verbose=2,shuffle=True,callbacks=[early_stopping_monitor])
                model.save("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/model_%s.h5"%(name))
                print(type(history.history["loss"]))
                history_dict = {"loss":history.history["loss"],"val_loss":history.history["val_loss"]}

                print(history_dict)
                print(type(history_dict))
                with open("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/history_%s.json"%name, "w") as f:
                    json.dump(history_dict, f)

    else:
        print("Start predict")
        model=load_trained_model("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/model_%s.h5"%name)



    print("Start predict_pointsName")
    if (useXGB):
        predictions=model.predict(xgb.DMatrix(X_train_scaled))
        predictions_test=model.predict(xgb.DMatrix(X_test_scaled)) 
    else:
        predictions=model.predict(X_train_scaled)
        predictions_test=model.predict(X_test_scaled)

    fpr, tpr, thresholds = metrics.roc_curve(y_train, predictions)
    roc_auc = metrics.auc(fpr,tpr)

    fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, predictions_test)
    roc_auc_test = metrics.auc(fpr_test,tpr_test)

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
        # 'Signal_Mass':X_train[:,33].tolist(),
        'Signal_Mass':X_train[:,-2].tolist(),
        # 'Scaled_Mass':X_train_scaled[:,33].tolist()
        'Scaled_Mass':X_train_scaled[:,-2].tolist()
        # 'Ymass':X_train_scaled[:,0].tolist()
    }
    data_test = {
        'predictions_test': predictions_test,
        'test_tag':y_test.tolist(),
        'weights':testweights.tolist(),
        # 'Signal_Mass':X_test[:,33].tolist(),
        'Signal_Mass':X_test[:,-2].tolist(),
        # 'Scaled_Mass':X_test_scaled[:,33].tolist()
        'Scaled_Mass':X_test_scaled[:,-2].tolist()

        # 'Ymass':X_test_scaled[:,0].tolist()
    }
    roc_train = {
        'train_fpr':fpr.tolist(),
        'train_tpr':tpr.tolist(),
        'thresholds':thresholds.tolist(),
        'thresholds_test':thresholds_test.tolist(),
        'test_fpr':fpr_test.tolist(),
        'test_tpr':tpr_test.tolist(),
        'auc':roc_auc.tolist(),
        'auc_test':roc_auc_test.tolist(),
    }

    with open("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/predictions_%s.json"%name, "w") as f:
        json.dump(roc_train, f)
    df_train=pd.DataFrame(data_train)
    df_train.to_csv("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/df_train_%s.csv"%name)
    df_test=pd.DataFrame(data_test)
    df_test.to_csv("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/df_test_%s.csv"%name)

    if compute_importance:

        results=[]
        print('computing feature importance....')

        oof_preds=model.predict(X_test_scaled,verbose=0).squeeze()
        baseline_mae = np.mean(np.abs(oof_preds-y_test))
        results.append({'feature':'BASELINE','mae':baseline_mae})

        for k in tqdm(range(len(training_columns)-1)):

            save_col = X_test_scaled[:,k].copy()
            np.random.shuffle(X_test_scaled[:,k])
                            
            # COMPUTE OOF MAE WITH FEATURE K SHUFFLED
            oof_preds = model.predict(X_test_scaled, verbose=0).squeeze() 
            mae = np.mean(np.abs( oof_preds-y_test ))
            results.append({'feature':training_columns[k+1],'mae':mae})
            X_test_scaled[:,k] = save_col            

        df = pd.DataFrame(results)
        df = df.sort_values('mae')
        plt.figure(figsize=(30,20))
        plt.barh(np.arange(len(training_columns)),df.mae)
        plt.yticks(np.arange(len(training_columns)),df.feature.values,fontsize=18)
        plt.title(' Feature Importance',size=18)
        plt.ylim((-1,len(training_columns)))
        plt.plot([baseline_mae,baseline_mae],[-1,len(training_columns)], '--', color='orange',
            label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
        plt.xlabel( 'OOF MAE with feature permuted',size=18)
        plt.ylabel('Feature',size=18)
        plt.legend()
        
        plt.savefig("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/feature_2.png")
        plt.show()
        df = df.sort_values('mae',ascending=False)
        df.to_csv('/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/feature_importance2.csv',index=False)

    #draw importance for each mass point
    # mass_list=[250, 260, 270, 280, 300, 320, 350, 400, 450,550, 600, 650, 700, 750, 800, 850, 900, 1000]
    mass_list=[250, 300, 400, 450,550, 650, 750, 900]

    con_df_test=pd.concat([testcsv,df_test],axis=1)
    con_df_train=pd.concat([traincsv,df_train],axis=1)
    for i in mass_list:
        results=[]
        selection="Signal_Mass=="+str(i)
        sig_df=con_df_test.query(selection)
        print("select %s events from signal%s"%(len(sig_df),i))
        # X_test_scaled=sig_df.values[:,:35]
        X_test_scaled=sig_df.values[:,:(len(training_columns)-1)]
        y_test=sig_df['test_tag']
        oof_preds_pnn=model.predict(X_test_scaled,verbose=0).squeeze()
        baseline_mae = np.mean(np.abs(oof_preds_pnn-y_test))
        results.append({'feature':'BASELINE','mae':baseline_mae})

        # for k in tqdm(range(35)):
        for k in tqdm(range(len(training_columns)-1)):

            save_col = X_test_scaled[:,k].copy()
            np.random.shuffle(X_test_scaled[:,k])
            
            # COMPUTE OOF MAE WITH FEATURE K SHUFFLED
            oof_preds_pnn = model.predict(X_test_scaled, verbose=0).squeeze() 
            mae = np.mean(np.abs( oof_preds_pnn-y_test ))
            results.append({'feature':training_columns[k+1],'mae':mae})
            X_test_scaled[:,k] = save_col 
        df = pd.DataFrame(results)
        df = df.sort_values('mae')
        plt.figure(figsize=(20,18))
        plt.barh(np.arange(len(training_columns)),df.mae)
        plt.yticks(np.arange(len(training_columns)),df.feature.values,fontsize=18)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.title(' Feature Importance',size=38)
        plt.ylim((-1,len(training_columns)))
        plt.plot([baseline_mae,baseline_mae],[-1,len(training_columns)], '--', color='orange',
            label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
        plt.xlabel( 'OOF MAE with feature permuted',size=28)
        plt.ylabel('Feature',size=28)
        plt.legend(fontsize=18)

        plt.savefig("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/sigmass"+str(i)+"feature.png")
        plt.show()
        df = df.sort_values('mae',ascending=False)
        df.to_csv("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/sigmass"+str(i)+"feature_importance.csv",index=False)   


    for i in mass_list:
        selection="Signal_Mass=="+str(i)
        sig_df_test=con_df_test.query(selection)
        sig_df_train=con_df_train.query(selection)
        plt.figure(figsize=(10,10))

        plt.hist(sig_df_test.query("test_tag==0").predictions_test,log=False,label="bkg",bins=np.linspace(0,1,20),weights=sig_df_test.query("test_tag==0").weights)
        plt.hist(sig_df_test.query("test_tag==1").predictions_test,log=False,label="signal",bins=np.linspace(0,1,20),alpha=0.7,weights=sig_df_test.query("test_tag==1").weights)
        plt.legend(fontsize=18)
        plt.savefig("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/sigmass"+str(i)+"_test_distributions.png")
        plt.show()
        plt.figure(figsize=(10,10))

        plt.hist(sig_df_train.query("train_tag==0").predictions,log=False,label="bkg",bins=np.linspace(0,1,20),weights=sig_df_train.query("train_tag==0").weights)
        plt.hist(sig_df_train.query("train_tag==1").predictions,log=False,label="signal",bins=np.linspace(0,1,20),alpha=0.7,weights=sig_df_train.query("train_tag==1").weights)
        plt.legend(fontsize=18)
        plt.savefig("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/sigmass"+str(i)+"_train_distributions.png")
        plt.show()

        background_test=sig_df_test.query("test_tag==0")
        bkgweight = background_test['weights'].values
        bkgpredic = background_test['predictions_test'].values
        #计算并存储sig bkg eff
        background_efficiencies_test = []
        signal_efficiencies_test = []

        background_train=sig_df_train.query("train_tag==0")
        bkgweight = background_train['weights'].values
        bkgpredic = background_train['predictions'].values
        #计算并存储sig bkg eff
        background_efficiencies_train = []
        signal_efficiencies_train = []

        for cut in np.linspace(0, 1, 100):
            sig_df_test['prediction_label'] = np.where(sig_df_test['predictions_test'] >= cut, 1, 0)
            background_efficiency = (sig_df_test[sig_df_test['test_tag'] == 0]['prediction_label'] * sig_df_test[sig_df_test['test_tag'] == 0]['weights']).sum() / sig_df_test[sig_df_test['test_tag'] == 0]['weights'].sum()
            signal_efficiency = (sig_df_test[sig_df_test['test_tag'] == 1]['prediction_label'] * sig_df_test[sig_df_test['test_tag'] == 1]['weights']).sum() / sig_df_test[sig_df_test['test_tag'] == 1]['weights'].sum()
            background_efficiencies_test.append(background_efficiency)
            signal_efficiencies_test.append(signal_efficiency)
        fpr_test, tpr_test, thresholds_test = roc_curve(sig_df_test['test_tag'], sig_df_test['predictions_test'], sample_weight=sig_df_test['weights'])
        # 计算 ROC 曲线下面积
        # auc = roc_auc_score(sig_df['test_tag'], sig_df['predictions_test'], sample_weight=sig_df['weights'])
        sorted_index = np.argsort(fpr_test)
        fpr_list_sorted =  np.array(fpr_test)[sorted_index]
        tpr_list_sorted = np.array(tpr_test)[sorted_index]
        auc_test=integrate.trapz(y=tpr_list_sorted, x=fpr_list_sorted)
        
        for cut in np.linspace(0, 1, 100):
            sig_df_train['prediction_label'] = np.where(sig_df_train['predictions'] >= cut, 1, 0)
            background_efficiency = (sig_df_train[sig_df_train['train_tag'] == 0]['prediction_label'] * sig_df_train[sig_df_train['train_tag'] == 0]['weights']).sum() / sig_df_train[sig_df_train['train_tag'] == 0]['weights'].sum()
            signal_efficiency = (sig_df_train[sig_df_train['train_tag'] == 1]['prediction_label'] * sig_df_train[sig_df_train['train_tag'] == 1]['weights']).sum() / sig_df_train[sig_df_train['train_tag'] == 1]['weights'].sum()
            background_efficiencies_train.append(background_efficiency)
            signal_efficiencies_train.append(signal_efficiency)
        fpr_train, tpr_train, thresholds_train = roc_curve(sig_df_train['train_tag'], sig_df_train['predictions'], sample_weight=sig_df_train['weights'])
        # 计算 ROC 曲线下面积
        # auc = roc_auc_score(sig_df['train_tag'], sig_df['predictions'], sample_weight=sig_df['weights'])
        sorted_index = np.argsort(fpr_train)
        fpr_list_sorted =  np.array(fpr_train)[sorted_index]
        tpr_list_sorted = np.array(tpr_train)[sorted_index]
        auc_train=integrate.trapz(y=tpr_list_sorted, x=fpr_list_sorted)

        plt.figure(figsize=(10,10))
        # 绘制 ROC 曲线
        plt.plot(background_efficiencies_test, signal_efficiencies_test, label='Test ROC curve %s'%auc_test)
        plt.plot(background_efficiencies_train, signal_efficiencies_train, label='Train ROC curve %s'%auc_train)
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random classifier')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Background efficiency',size=24)
        plt.ylabel('Signal efficiency',size=24)
        plt.title('ROC curve', fontdict={'fontsize': 28})
        plt.legend(fontsize=18)
        plt.savefig("/hpcfs/cms/cmsgpu/shaoweisong/DNN/cat7/sigmass"+str(i)+"trainROC.png")
        plt.show()
    




