"""
author: Athina Apostolelli

This script is used to retrain the 1D CNN model used for SWR detection, aiming to optimize the accuracy of detection 
using LFP that has been filtered and downsampled to 2000 Hz. 
The CNN used for SWR detection was developed by the Prida lab (https://github.com/PridaLab/rippl-AI).

The data that are used to retrain are from Eminhan Ozil's dataset (two hemispheres).
"""
#%%
import numpy as np
import os
import sys
import scipy
import scipy.io
import keras
from tqdm import tqdm
import matplotlib.pyplot as plt

rippl_AI_repo = 'C:/Users/RECORDING/Athina/Github/rippl-AI'
ripple_analysis_dir = 'C:/Users/RECORDING/Athina/Ripples/'
sys.path.insert(0, rippl_AI_repo)
sys.path.insert(1,'C:/Users/RECORDING/Athina/Github/rippl-AI/optimized_models')
import rippl_AI
import aux_fcn
from importlib import reload
reload(rippl_AI)
reload(aux_fcn)

from downsample_filter_LFP import import_lfp

#%% 
def get_performance_after_training(save_path, arch, LFP_val, SWR_val, d_sf=2000):
    # Load model 
    model = keras.models.load_model(save_path)

    # The correct n_channels and timesteps needs to be passed to predict for the fcn to work when using new_model
    if arch=='XGBOOST':
        n_channels=8
        timesteps=16
    elif arch=='SVM':
        n_channels=8
        timesteps=1
    elif arch=='LSTM':
        n_channels=8
        timesteps=32
    elif arch=='CNN2D':
        n_channels=8
        timesteps=40
    elif arch=='CNN1D':
        n_channels=8
        timesteps=16
    
    val_LFP_array = np.array([])  
    for LFP in LFP_val:
        # 1st session in the array
        print('Original training data shape: ',LFP.shape)
        if val_LFP_array.size == 0:
            val_LFP_array = aux_fcn.process_LFP(LFP, sf=20000, d_sf=d_sf, channels=np.arange(0,8))
            offset = len(val_LFP_array)/d_sf
        # Append the rest of the sessions, taking into account the length (in seconds) 
        # of the previous sessions, to cocatenate the events' times
        else:
            aux_LFP = aux_fcn.process_LFP(LFP, sf=20000, d_sf=d_sf, channels=np.arange(0,8))
            val_LFP_array = np.vstack([val_LFP_array,aux_LFP])
            offset += len(aux_LFP)/d_sf

    # Get SWR probability 
    SWR_prob = aux_fcn.prediction_parser(val_LFP_array, new_model=model, n_channels=n_channels, n_timesteps=timesteps)  

    # Get SWR time intervals   
    predictions = rippl_AI.get_intervals(y=SWR_prob, threshold=0.8, LFP_norm=val_LFP_array, sf=d_sf, win_size=100, \
                  file_path=os.path.join(save_path, f"predictions_{os.path.basename(save_path)}.txt"), merge_win=128)
    
    # Get performance
    performance = aux_fcn.get_performance(pred_events=predictions, true_events=SWR_val, threshold=0.8)[0:3]

    with open(os.path.join(save_path, f"performance_{os.path.basename(save_path)}.txt"), 'w') as f:
        f.write('\t'.join(['precision', 'recall', 'F1']) + '\n')
        f.write('\t'.join(map(str, performance)) + '\n')


        param_search_rippl_AI


def param_search_rippl_AI(x_training, GT_training, conf, save_path, sf=2000):
    th_arr=np.linspace(0.1,0.9,9)
    model_name_arr=[]           # To plot in the next cell
    model_arr=[]                # Actual model array, used in the next validation section
    n_channels=x_training.shape[1]
    timesteps_arr=conf['timesteps']

    config_arr=conf['configuration']
    epochs_arr=conf['epochs']
    train_batch_arr=conf['train_batch']                                               

    l_ts = len(timesteps_arr)
    l_conf = len(config_arr)
    l_epochs = len(epochs_arr)
    l_batch = len(train_batch_arr)
    n_iters = l_ts*l_conf*l_epochs*l_batch
    # GT is in the shape (n_events x 2), a y output signal with the same length as x is required
    perf_train_arr=np.zeros(shape=(n_iters,len(th_arr),3)) # Performance array, (n_models x n_th x 3 ) [P R F1]
    perf_test_arr=np.zeros_like(perf_train_arr)
    timesteps_arr_ploting=[]            # Array that will be used in the validation, to be able to call the function predict

    print(f'{n_channels} channels will be used to train the CNN1D models')

    print(f'{n_iters} models will be trained')

    x_test_or,GT_test,x_train_or,GT_train=aux_fcn.split_data(x_training,GT_training,split=0.7,sf=sf)

    y_test_or= np.zeros(shape=(len(x_test_or)))
    for ev in GT_test:
        y_test_or[int(sf*ev[0]):int(sf*ev[1])]=1
    y_train_or= np.zeros(shape=(len(x_train_or)))
    for ev in GT_train:
        y_train_or[int(sf*ev[0]):int(sf*ev[1])]=1


    for i_ts,timesteps in enumerate(timesteps_arr):
        x_train=x_train_or[:len(x_train_or)-len(x_train_or)%timesteps].reshape(-1,timesteps,n_channels)
        y_train_aux=y_train_or[:len(y_train_or)-len(y_train_or)%timesteps].reshape(-1,timesteps)
        x_test=x_test_or[:len(x_test_or)-len(x_test_or)%timesteps].reshape(-1,timesteps,n_channels)
        y_test_aux=y_test_or[:len(y_test_or)-len(y_test_or)%timesteps].reshape(-1,timesteps)

        y_train=np.zeros(shape=[x_train.shape[0],1])
        for i in range(y_train_aux.shape[0]):
            y_train[i]=1  if any (y_train_aux[i]==1) else 0
        print("Train Input and Output dimension", x_train.shape,y_train.shape)
        
        y_test=np.zeros(shape=[x_test.shape[0],1])
        for i in range(y_test_aux.shape[0]):
            y_test[i]=1  if any (y_test_aux[i]==1) else 0

        for i_conf, configuration in enumerate(config_arr):
            for i_epochs,epochs in enumerate(epochs_arr):
                for i_batch,train_batch in enumerate(train_batch_arr):
                    iter=((i_ts*l_conf+i_conf)*l_epochs + i_epochs)*l_batch + i_batch
                    print(f"\nIteration {iter+1} out of {n_iters}")
                    print(f'Number of channels: {n_channels:d}, Time steps: {timesteps:d},\nconfiguration: {configuration}\nEpochs: {epochs:d}, Samples per batch: {train_batch:d}')

                    model = aux_fcn.build_CNN1D(n_channels,timesteps,configuration)
                    # Training
                    model.fit(x_train, y_train, shuffle=False, epochs=epochs, batch_size=train_batch, validation_data=(x_test,y_test), verbose=1)
                    model_arr.append(model)
                    # Prediction
                    test_signal = model.predict(x_test,verbose=1)
                    train_signal=model.predict(x_train,verbose=1)

                    y_train_predict=np.empty(shape=(x_train.shape[0]*timesteps,1,1))
                    for i,window in enumerate(train_signal):
                        y_train_predict[i*timesteps:(i+1)*timesteps]=window
                    y_test_predict=np.empty(shape=(x_test.shape[0]*timesteps,1,1))
                    for i,window in enumerate(test_signal):
                        y_test_predict[i*timesteps:(i+1)*timesteps]=window

                    ############################
                    for i,th in enumerate(th_arr):
                        # Test
                        ytest_pred_ind=aux_fcn.get_predictions_index(y_test_predict,th)/sf
                        perf_test_arr[iter,i]=aux_fcn.get_performance(ytest_pred_ind,GT_test,0)[0:3]
                        # Train
                        ytrain_pred_ind=aux_fcn.get_predictions_index(y_train_predict,th)/sf
                        perf_train_arr[iter,i]=aux_fcn.get_performance(ytrain_pred_ind,GT_train,0)[0:3]

                        # Saving the model
                        model_name=f"CNN1D_Ch{n_channels:d}_Ts{timesteps:03d}_C{i_conf:02d}_E{epochs:02d}_TB{train_batch:04d}"
                        model.save(os.path.join(save_path,'explore_models',model_name))

                        model_name_arr.append(model_name)
                        timesteps_arr_ploting.append(timesteps)
    return

# =================================================
#                  SET PARAMETERS
# =================================================
#%%
if __name__ == "__main__": 

    sf = 20000
    downsampled_fs = 2000
    arch = 'CNN1D'
    training_batch = 32
    epochs = 10
    param_search = True

    save_path = os.path.join(ripple_analysis_dir, 'retraining', 'new_rippl_AI', 'model_' + arch + '_' + str(epochs) + 'epochs_' + str(training_batch) + 'bs_')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #%%
    # =================================================
    #                   TRAINING DATA
    # =================================================
    # Define training sessions
    train_sessions = ['6_240205_150311', '9_240229_163045', '128ch_concatenated_sessions']
    train_animals = ['rEO_06'] * len(train_sessions)
    train_channels = []
    for s in range(len(train_sessions)):
        if train_animals[s] == 'rEO_06':
            HPC_channels = [101,98,93,92,95,94,81,110,27,26,29,37,28,34,35,31]
        elif train_animals[s] == 'rEO_05':
            HPC_channels = [107,106,83,109,108,86,105,89,39,27,36,26,37,29,34,35]
        elif train_animals[s] == 'rEO_07':
            HPC_channels = [89,102,88,103,90,101,98,95,36,26,29,15,35,46,47,49]

        train_channels.append(HPC_channels)
        
    # train_ch_map = []

    # Load training LFPs - takes a long time
    train_LFPs = []
    for s, session in enumerate(train_sessions):
        raw_data_dir = os.path.join('D:/Rat_Recording', train_animals[s])

        if 'concatenated' in session:
            data_file = os.path.join(raw_data_dir, session, '128ch_concat_data.dat')
        else:
            data_file = os.path.join(raw_data_dir, session, 'amplifier.dat')
        # lfp_file = os.path.join(ripple_analysis_dir, 'lfp_data', animal, session + '.amplifier_ds.lfp')

        lfp_data = import_lfp(raw_data_file=data_file, num_channels=128, channels=train_channels[s], sample_rate=sf, verbose=False)
        lfp = np.transpose(lfp_data)  # n_samples x n_channels
        
        train_LFPs.append(lfp[:,0:8])
        train_LFPs.append(lfp[:,8:16])
        
        # train_ch_map.append([1,-1,-1,2,-1,-1,-1,4]) # left hemi 
        # train_ch_map.append([0,-1,-1,3,-1,-1,-1,5]) # right hemi

    # Load training SWRs
    train_SWRs = []
    for s, session in enumerate(train_sessions):
        ripple_file = os.path.join(ripple_analysis_dir, 'curated_ripples', train_animals[s], session, 'curated_ripples.mat')

        ripple_data = scipy.io.loadmat(ripple_file)
        ripple_classes = [label[0] for label in ripple_data['ripple_classes'].flatten()]
        ripple_timestamps = ripple_data['ripple_timestamps']
                
        left_indices = [i for i, label in enumerate(ripple_classes) if 'left' in label]
        right_indices = [i for i, label in enumerate(ripple_classes) if 'right' in label]

        train_SWRs.append(ripple_timestamps[left_indices,:]) 
        train_SWRs.append(ripple_timestamps[right_indices,:]) 

    #%%
    # =================================================
    #                  VALIDATION DATA
    # =================================================
    # Define validation sessions
    val_sessions = ['3_240112_150230', '4_240130_152039', '5_240202_120214']
    val_animals = ['rEO_06'] * len(val_sessions)
    # val_sessions = ['128ch_concatenated_sessions', 'session_1_2_230904_173347']
    # val_animals = ['rEO_05', 'rEO_07']
    val_channels = []
    for s in range(len(val_sessions)):
        if val_animals[s] == 'rEO_06':
            HPC_channels = [101,98,93,92,95,94,81,110,27,26,29,37,28,34,35,31]
        elif val_animals[s] == 'rEO_05':
            HPC_channels = [107,106,83,109,108,86,105,89,39,27,36,26,37,29,34,35]
        elif val_animals[s] == 'rEO_07':
            HPC_channels = [89,102,88,103,90,101,98,95,36,26,29,15,35,46,47,49]

        val_channels.append(HPC_channels)

    # val_ch_map = []

    # Load validation LFPs 
    val_LFPs = []
    for s, session in enumerate(val_sessions):
        raw_data_dir = os.path.join('D:/Rat_Recording', val_animals[s])

        if 'concatenated' in session:
            data_file = os.path.join(raw_data_dir, session, '128ch_concat_data.dat')
        else:
            data_file = os.path.join(raw_data_dir, session, 'amplifier.dat')
        # lfp_file = os.path.join(ripple_analysis_dir, 'lfp_data', animal, session + '.amplifier_ds.lfp')

        lfp_data = import_lfp(raw_data_file=data_file, num_channels=128, channels=val_channels[s], sample_rate=sf, verbose=False)
        lfp = np.transpose(lfp_data)  # n_samples x n_channels
        
        val_LFPs.append(lfp[:,0:8])
        val_LFPs.append(lfp[:,8:16])
        
        # val_ch_map.append([1,-1,-1,2,-1,-1,-1,4]) # left hemi 
        # val_ch_map.append([0,-1,-1,3,-1,-1,-1,5]) # right hemi

    # Load validation SWRs
    val_SWRs = []
    for s, session in enumerate(val_sessions):
        ripple_file = os.path.join(ripple_analysis_dir, 'curated_ripples', val_animals[s], session, 'curated_ripples.mat')

        ripple_data = scipy.io.loadmat(ripple_file)
        ripple_classes = [label[0] for label in ripple_data['ripple_classes'].flatten()]
        ripple_timestamps = ripple_data['ripple_timestamps']
                
        left_indices = [i for i, label in enumerate(ripple_classes) if 'left' in label]
        right_indices = [i for i, label in enumerate(ripple_classes) if 'right' in label]

        val_SWRs.append(ripple_timestamps[left_indices,:])
        val_SWRs.append(ripple_timestamps[right_indices,:])  
        

    # =================================================
    #                       TRAIN
    # ================================================= 
    #%%
    if param_search:
        retrain_LFPs, retrain_SWRs, norm_val_LFP, val_SWRs = rippl_AI.prepare_training_data(train_LFPs, train_SWRs, val_LFPs, val_SWRs, sf=sf, d_sf=downsampled_fs, channels=np.arange(0,8))
        
        conf= {"timesteps":   [16, 32],        
            "configuration":  [[4,2],[2,1],[8,2],[4,1],[16,2],[8,1],[32,2]],  
            "epochs":      [1, 5, 10],         # 1, 2, 3, 5...
            "train_batch": [2**5],      # 32, 64, 128...
        }
        param_search_rippl_AI(x_training=retrain_LFPs, GT_training=retrain_SWRs, conf=conf, save_path=save_path, sf=downsampled_fs)
  
    else:
        params = {'Epochs': epochs, 'Training batch': training_batch}

        retrain_LFPs, retrain_SWRs, norm_val_LFP, val_SWRs = rippl_AI.prepare_training_data(train_LFPs, train_SWRs, val_LFPs, val_SWRs, sf=sf, d_sf=downsampled_fs, channels=np.arange(0,8))

        current_dir = os.getcwd()
        os.chdir(rippl_AI_repo)
        rippl_AI.retrain_model(retrain_LFPs, retrain_SWRs, norm_val_LFP, val_SWRs, arch='CNN1D', parameters=params, d_sf=downsampled_fs, save_path=save_path, merge_win=0)
        os.chdir(current_dir)

    #%%
    # get_performance_after_training(save_path, arch='CNN1D', LFP_val=val_LFPs, SWR_val=val_SWRs, d_sf=2000)


    # retrain_LFPs, retrain_SWRs, norm_val_LFP, val_SWRs = prepare_training_data_new(train_LFPs,train_SWRs,val_LFPs,val_SWRs,train_ch_map,val_ch_map,sf=sf,downsampled_fs=downsampled_fs)

    # current_dir = os.getcwd()
    # os.chdir(rippl_AI_repo)
    # retrain_model_new(retrain_LFPs, retrain_SWRs, norm_val_LFP, val_SWRs, arch=arch, sf=sf, parameters=params, save_path=save_path)
    # os.chdir(current_dir)
# %%
