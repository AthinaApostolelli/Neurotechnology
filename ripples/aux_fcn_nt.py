"""
author: Athina Apostolelli

This script contains auxiliary functions used for SWR detection using the CNN developed by the Prida lab (https://github.com/PridaLab/rippl-AI).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('fast')
import os
import sys
import keras

from tensorflow import keras
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler

rippl_AI_repo = 'C:/Users/RECORDING/Athina/Github/rippl-AI'
ripple_analysis_dir = 'C:/Users/RECORDING/Athina/Ripples/'
sys.path.insert(1, rippl_AI_repo)
import rippl_AI
import aux_fcn
from importlib import reload
reload(rippl_AI)
reload(aux_fcn)


def process_LFP_new(LFP, sf, downsampled_fs, channels):
    ''' 
	This function processes the LFP before calling the detection algorithm.
	1. It extracts the desired channels from the original LFP, and interpolates where there is a value of -1.
	2. Downsamples the LFP to the specified downsampled_fs Hz.
	3. Normalizes each channel separately by z-scoring them.

	Mandatory inputs:
		LFP: 		LFP recorded data (np.array: n_samples x n_channels).
		sf: 		Original sampling frequency (in Hz).
        downsampled_fs: Frequency at which to downsample the data 
		channels: 	channel to which compute the undersampling and z-score normalization. Counting starts in 0. 
					If channels contains any -1, interpolation will be also applied. 
					See channels of rippl_AI.predict(), or aux_fcn.interpolate_channels() for more information.
	Output:
		LFP_norm: normalized LFP (np.array: n_samples x len(channels)). It is undersampled to 1250Hz, z-scored, 
					and transformed to used the channels specified in channels.
    '''
    data = aux_fcn.interpolate_channels(LFP,channels)
    if sf!=downsampled_fs:
        print("Downsampling data from %d Hz to %d Hz..."%(sf, downsampled_fs), end=" ")
        data = aux_fcn.downsample_data(data, sf, downsampled_fs=downsampled_fs)
        print("Shape of downsampled data:",data.shape)
    else:
        print("Data is already sampled at %d Hz!"%(sf))
	
    print('Normalizing data...')
    normalized_data = aux_fcn.z_score_normalization(data)

    print("Shape of loaded data after downsampling and z-score: ", np.shape(normalized_data))
    return normalized_data

# def prepare_training_data_new(train_LFPs,train_GTs,val_LFPs,val_GTs,sf=30000,downsampled_fs=2000,channels=np.arange(0,8)):
def prepare_training_data_new(train_LFPs,train_GTs,val_LFPs,val_GTs,train_ch_map,val_ch_map,sf=30000,downsampled_fs=2000):
    '''
        Prepares data for training: subsamples, interpolates (if required), z-scores and concatenates 
        the train/test data passed. Does the same for the validation data, but without concatenating
        inputs:
            train_LFPs:  (n_train_sessions) list with the raw LFP of n sessions that will be used to train
            train_GTs:   (n_train_sessions) list with the GT events of n sessions, in the format [ini end] in seconds
            (A): quizá se podría quitar esto, lo de formatear tambien las de validacion 
            val_LFPs:    (n_val_sessions) list: with the raw LFP of the sessions that will be used in validation
            val_GTs:     (n_val_sessions) list: with the GT events of n validation sessions
            sf:          (int) original sampling frequency of the data TODO (consultar con Andrea): make it an array, so every session could have a different sf
			downsampled_fs (int) frequency to downsample the data 
            channels:    (n_channels) np.array. Channels that will be used to generate data. Check interpolate_channels for more information
        output:
            retrain_LFP: (n_samples x n_channels): sumbsampled, z-scored, interpolated and concatenated data from all the training sessions
            retrain_GT:  (n_events x 2): concatenation of all the events in the training sessions
            norm_val_GT: (n_val_sessions) list: list with the normalized LFP of all the val sessions
            val_GTs:     (n_val_sessions) list: Gt events of each val sessions
    A Rubio LCN 2023

    '''
    assert len(train_LFPs) == len(train_GTs), "The number of train LFPs doesn't match the number of train GTs"
    assert len(val_LFPs) == len(val_GTs), "The number of test LFPs doesn't match the number of test GTs"

    # All the training sessions data and GT will be concatenated in one data array and one GT array (2 x n events)
    retrain_LFP=[]
    for i, (LFP,GT,channels) in enumerate(zip(train_LFPs,train_GTs,train_ch_map)):
        # 1st session in the array
        print('Original training data shape: ',LFP.shape)
        if i == 0:
        # if retrain_LFP==[]:
            retrain_LFP=process_LFP_new(LFP,sf,downsampled_fs,channels)
            offset=len(retrain_LFP)/downsampled_fs
            retrain_GT=GT
        # Append the rest of the sessions, taking into account the length (in seconds) 
        # of the previous sessions, to cocatenate the events' times
        else:
            aux_LFP=process_LFP_new(LFP,sf,downsampled_fs,channels)
            retrain_LFP=np.vstack([retrain_LFP,aux_LFP])
            retrain_GT=np.vstack([retrain_GT,GT+offset])
            offset+=len(aux_LFP)/downsampled_fs
    # Each validation session LFP will be normalized, etc and stored in an array
    #  the GT needs no further treatment
    norm_val_LFP=[]
    for (LFP,channels) in zip(val_LFPs,val_ch_map):
        print('Original validation data shape: ',LFP.shape)
        norm_val_LFP.append(process_LFP_new(LFP,sf,downsampled_fs,channels))

    return retrain_LFP, retrain_GT, norm_val_LFP, val_GTs

def retraining_parser_new(arch,x_train_or,events_train,x_test,events_test,sf,params=None):
	'''
	[model,y_train,y_test] = retraining_parser(arch,x_train_or,events_train,x_test,events_test,params=None)\n
    Performs the retraining of the best model of the desired architecture\n
    Inputs:
		arch:			string, with the desired architecture model to be retrained
		x_train_or:		[n train samples x 8], normalized LFP that will be used to retrain the model
		events_train: 	[n train events x 2], begin and end timess of the train events
		x_test_or:		[n test samples x 8], normalized LFP that will be used to retrain the model
		events_train: 	[n test events x 2], begin and end timess of the test events 
		sf:             sampling frequency (added by Athina)
		
		Optional inputs
			params: dictionary, with the parameters that will be use in each specific architecture retraining
			- In 'XGBOOST': not needed
			- In 'SVM':     
				params['Undersampler proportion']. Any value between 0 and 1. This parameter eliminates 
								samples where no ripple is present untill the desired proportion is achieved: 
								Undersampler proportion= Positive samples/Negative samples
			- In 'LSTM', 'CNN1D' and 'CNN2D': 
				params['Epochs']. The number of times the training data set will be used to train the model
				params['Training batch']. The number of windows that will be processed before updating the weights   
	Output:		
    	model: The retrained model
	    y_train_p: [n_train_samples], output of the model using the training data
	    y_test_p:  [n_test_samples], output of the model using the test data
	A Rubio LCN 2023
	'''
	global model 

	# Input data preparing for training
	x_train = np.copy(x_train_or)
	# Sampling frequency hard fixed to 1250 - fixed by Athina
	y_train= np.zeros(shape=(len(x_train)))
	for ev in events_train:
		y_train[int(sf*ev[0]):int(sf*ev[1])]=1

	y_test= np.zeros(shape=(len(x_test)))
	for ev in events_test:
		y_test[int(sf*ev[0]):int(sf*ev[1])]=1

	x_train_len=x_train.shape[0]
	x_test_len=x_test.shape[0]
	
	# Automatically hard coded to input the required shape for the best model of each arch 
	if arch=='XGBOOST':
		n_channels=8
		timesteps=16
		# Making the input data and expected output compatible with he resizing
		x_train=x_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps*n_channels)
		y_train_aux=y_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps)
		y_train=aux_fcn.rec_signal(y_train_aux)

		x_test=x_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps*n_channels)
		y_test_aux=y_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps)
		y_test=aux_fcn.rec_signal(y_test_aux)
		# model load
		model=XGBClassifier()
		model.load_model(os.path.join('optimized_models','XGBOOST_1_Ch8_W60_Ts016_D7_Lr0.10_G0.25_L10_SCALE1'))

		model.fit(x_train, y_train,verbose=True,eval_set = [(x_test, y_test)])
		
		y_train_p=np.zeros(shape=(x_train_len,1,1))
		train_signal=model.predict_proba(x_train)[:,1]
		for i,window in enumerate(train_signal):
			y_train_p[i*timesteps:(i+1)*timesteps]=window
		y_test_p=np.zeros(shape=(x_test_len,1,1))	
		test_signal=model.predict_proba(x_test)[:,1]
		for i,window in enumerate(test_signal):
			y_test_p[i*timesteps:(i+1)*timesteps]=window
	elif arch=='SVM':
		n_channels=8
		timesteps=1
		# Making the input data and expected output compatible with he resizing
		x_train=x_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps*n_channels)
		y_train_aux=y_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps)
		y_train=aux_fcn.rec_signal(y_train_aux)

		x_test=x_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps*n_channels)
		y_test_aux=y_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps)
		y_test=aux_fcn.rec_signal(y_test_aux)

		#Under sampler: discards windows where there is no ripples untill the desired proportion between ripple/no ripple is achieved
		# If no params is provided, the defect proportion will be 0.5
		if params==None:
			us_prop=0.5
		else:
			us_prop=params['Unsersampler proportion']
		rus = RandomUnderSampler(sampling_strategy=us_prop)
		x_train_us, y_train_us = rus.fit_resample(x_train, y_train)
		
		print(f"Under sampling result: {x_train_us.shape}")
		# model load
		model=aux_fcn.fcn_load_pickle(os.path.join('optimized_models','SVM_1_Ch8_W60_Ts001_Us0.05'))
		# model fit
		model=model.fit(x_train_us, y_train_us)

		y_train_p=np.zeros(shape=(x_train_len,1,1))
		train_signal=model.predict_proba(x_train)[:,1]
		for i,window in enumerate(train_signal):
			y_train_p[i*timesteps:(i+1)*timesteps]=window
		y_test_p=np.zeros(shape=(x_test_len,1,1))	
		test_signal=model.predict_proba(x_test)[:,1]
		for i,window in enumerate(test_signal):
			y_test_p[i*timesteps:(i+1)*timesteps]=window
	elif arch=='LSTM':
		n_channels=8
		timesteps=32
		x_train=x_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps,n_channels)
		y_train=y_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps,1)
		x_test=x_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps,n_channels)
		y_test=y_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps,1)
		print("Input and output shape: ",x_train.shape,y_train.shape)
		model = keras.models.load_model(os.path.join('optimized_models','LSTM_1_Ch8_W60_Ts32_Bi0_L4_U11_E10_TB256'))
		# If no parameters are provided, 5 epochs and 32 as training batch will be used
		if params==None:
			epochs=5
			tb=32
		else:
			epochs=params['Epochs']
			tb=params['Training batch']
		model.fit(x_train, y_train, epochs=epochs,batch_size=tb,validation_data=(x_test,y_test), verbose=1)
	    
		y_train_p = model.predict(x_train,verbose=1)
		y_train_p=y_train_p.reshape(-1,1,1)
		y_test_p = model.predict(x_test,verbose=1)
		y_test_p=y_test_p.reshape(-1,1,1)
	elif arch=='CNN1D':
		n_channels=8
		timesteps=16
		x_train=x_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps,n_channels)
		y_train_aux=y_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps)
		x_test=x_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps,n_channels)
		y_test_aux=y_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps)

		y_train=np.zeros(shape=[x_train.shape[0],1])
		for i in range(y_train_aux.shape[0]):
			y_train[i]=1  if any (y_train_aux[i]==1) else 0
		print("Train Input and Output dimension", x_train.shape,y_train.shape)
		
		y_test=np.zeros(shape=[x_test.shape[0],1])
		for i in range(y_test_aux.shape[0]):
			y_test[i]=1  if any (y_test_aux[i]==1) else 0

		optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
		model = keras.models.load_model(os.path.join('optimized_models','CNN1D_1_Ch8_W60_Ts16_OGmodel12'), compile=False)
		model.compile(loss="binary_crossentropy", optimizer=optimizer)
		if params==None:
			epochs=20
			tb=32
		else:
			epochs=params['Epochs']
			tb=params['Training batch']
		model.fit(x_train, y_train,shuffle=False, epochs=epochs,batch_size=tb,validation_data=(x_test,y_test), verbose=1)
		y_train_p=np.zeros(shape=(x_train_len,1,1))
		train_signal=model.predict(x_train)
		for i,window in enumerate(train_signal):
			y_train_p[i*timesteps:(i+1)*timesteps]=window
		y_test_p=np.zeros(shape=(x_test_len,1,1))	
		test_signal=model.predict(x_test)
		for i,window in enumerate(test_signal):
			y_test_p[i*timesteps:(i+1)*timesteps]=window
	elif arch=='CNN2D':
		n_channels=8
		timesteps=40
		x_train=x_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps,n_channels,1)
		y_train_aux=y_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps,1)
		x_test=x_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps,n_channels,1)
		y_test_aux=y_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps,1)

		y_train=np.zeros(shape=[x_train.shape[0],1])
		for i in range(y_train_aux.shape[0]):
			y_train[i]=1  if any (y_train_aux[i]==1) else 0
		y_test=np.zeros(shape=[x_test.shape[0],1])
		for i in range(y_test_aux.shape[0]):
			y_test[i]=1  if any (y_test_aux[i]==1) else 0
	
		model = keras.models.load_model(os.path.join('optimized_models','CNN2D_1_Ch8_W60_Ts40_OgModel'))
		# If no parameters are provided, 20 epochs and 32 as training batch will be used
		if params==None:
			epochs=20
			tb=32
		else:
			epochs=params['Epochs']
			tb=params['Training batch']
		model.fit(x_train, y_train,shuffle=False, epochs=epochs,batch_size=tb,validation_data=(x_test,y_test), verbose=1)
		y_train_p=np.zeros(shape=(x_test_len,1,1))
		train_signal=model.predict(x_train)
		for i,window in enumerate(train_signal):
			y_train_p[i*timesteps:(i+1)*timesteps]=window
		y_test_p=np.zeros(shape=(x_test_len,1,1))	
		test_signal=model.predict(x_test)
		for i,window in enumerate(test_signal):
			y_test_p[i*timesteps:(i+1)*timesteps]=window
        
	return(model,y_train_p.reshape(-1),y_test_p.reshape(-1))

def retrain_model_new(LFP_retrain,GT_retrain,LFP_val,GT_val,arch,sf,parameters=None,save_path=None):
    '''
        Retrains the best model of the specified architecture with the retrain data and the specified parameters. Performs validation if validation data is provided, and plots the train, test and validation performance.
        inputs:
            LFP_retrain:  (n_samples x n_channels)  concatenated LFP of all the trained sessions
            GT_retrain:   (n_events x 2) list with the concatenated GT events times of n sessions, in the format [ini end] in seconds
            arch:         string, architecture of the model to be retrained
            sf:           sampling frequency 
            LFP_val:      (n_val_sessions) list: with the normalized LFP of the sessions that will be used in validation
            GT_val:       (n_val_sessions) list: with the GT events of the validation sessions
            Optional inputs
                parameters: dictionary, with the parameters that will be use in each specific architecture retraining
                - In 'XGBOOST': not needed
                - In 'SVM':     
                    parameters['Undersampler proportion']. Any value between 0 and 1. This parameter eliminates 
                                    samples where no ripple is present untill the desired proportion is achieved: 
                                    Undersampler proportion= Positive samples/Negative samples
                - In 'LSTM', 'CNN1D' and 'CNN2D': 
                    parameters['Epochs']. The number of times the training data set will be used to train the model
                    parameters['Training batch']. The number of windows that will be processed before updating the weights
                save_path: string, path where the retrained model will be saved
        output:
            retrain_LFP: (n_samples x n_channels): sumbsampled, z-scored, interpolated and concatenated data from all the training sessions
            retrain_GT:  (n_events x 2): concatenation of all the events in the training sessions
            norm_val_GT: (n_val_sessions) list: list with the normalized LFP of all the val sessions
            val_GTs:     (n_val_sessions) list: Gt events of each val sessions
    A Rubio LCN 2023

    '''
    # Do the train/test split. Feel free to try other proportions
    LFP_test,events_test,LFP_train,events_train = aux_fcn.split_data(LFP_retrain,GT_retrain,sf=sf,split=0.7)

    print(f'Number of validation sessions: {len(LFP_val)}') #TODO: for shwoing length and events
    print(f'Shape of train data: {LFP_train.shape}, Number of train events: {events_train.shape[0]}')
    print(f'Shape of test data: {LFP_test.shape}, Number of test events: {events_test.shape[0]}')

    # prediction parser returns the retrained model, the output predictions probabilities
    model,y_pred_train,y_pred_test = retraining_parser_new(arch,LFP_train,events_train,LFP_test,events_test,sf,params=parameters)

    # Save model if save_path is not empty
    if save_path:
        aux_fcn.save_model(model,arch,save_path)

    # Plot section #
    # for loop iterating over the validation data
    val_pred=[]
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
    
    for LFP in LFP_val:
        val_pred.append(aux_fcn.prediction_parser(LFP,arch=arch,new_model=model,n_channels=n_channels,n_timesteps=timesteps))
        # val_pred.append(rippl_AI.predict(LFP,sf=sf,arch=arch,new_model=model,n_channels=n_channels,n_timesteps=timesteps)[0])
    
    # Extract and plot the train and test performance
    th_arr=np.linspace(0.1,0.9,9)
    F1_train=np.empty(shape=len(th_arr))
    precision_train=np.empty(shape=len(th_arr))
    recall_train=np.empty(shape=len(th_arr))
    F1_test=np.empty(shape=len(th_arr))
    precision_test=np.empty(shape=len(th_arr))
    recall_test=np.empty(shape=len(th_arr))
    for i,th in enumerate(th_arr):
        pred_train_events = aux_fcn.get_predictions_index(y_pred_train,th)/sf
        pred_test_events = aux_fcn.get_predictions_index(y_pred_test,th)/sf
        precision_train[i],recall_train[i],F1_train[i],_,_,_ = aux_fcn.get_performance(pred_train_events,events_train,verbose=False)
        precision_test[i],recall_test[i],F1_test[i],_,_,_ = aux_fcn.get_performance(pred_test_events,events_test,verbose=False)
    

    fig,axs=plt.subplots(3,2,figsize=(12,12),sharey='all')
    axs[0,0].plot(th_arr,F1_train,'k.-')
    axs[0,0].plot(th_arr,F1_test,'b.-')
    axs[0,0].legend(['Train','Test'])
    axs[0,0].set_ylim([0 ,max(max(F1_train), max(F1_test)) + 0.1])
    axs[0,0].set_title('F1 test and train')
    axs[0,0].set_ylabel('F1')
    axs[0,0].set_xlabel('Threshold')
	
    axs[1,0].plot(th_arr,precision_train,'k.-')
    axs[1,0].plot(th_arr,precision_test,'b.-')
    axs[1,0].legend(['Train','Test'])
    axs[1,0].set_ylim([0 ,max(max(precision_train), max(precision_test)) + 0.1])
    axs[1,0].set_title('precision test and train')
    axs[1,0].set_ylabel('precision')
    axs[1,0].set_xlabel('Threshold')
	
    axs[2,0].plot(th_arr,recall_train,'k.-')
    axs[2,0].plot(th_arr,recall_test,'b.-')
    axs[2,0].legend(['Train','Test'])
    axs[2,0].set_ylim([0 ,max(max(recall_train), max(recall_test)) + 0.1])
    axs[2,0].set_title('recall test and train')
    axs[2,0].set_ylabel('recall')
    axs[2,0].set_xlabel('Threshold')


    # Validation plot in the second ax
    F1_val=np.zeros(shape=(len(LFP_val),len(th_arr)))
    precision_val=np.zeros(shape=(len(LFP_val),len(th_arr)))
    recall_val=np.zeros(shape=(len(LFP_val),len(th_arr)))
    for j,pred in enumerate(val_pred):
        for i,th in enumerate(th_arr):
            pred_val_events = aux_fcn.get_predictions_index(pred,th)/sf
            precision_val[j,i],recall_val[j,i],F1_val[j,i],_,_,_ = aux_fcn.get_performance(pred_val_events,GT_val[j],verbose=False)

    for i in range(len(LFP_val)):
        axs[0,1].plot(th_arr,F1_val[i])
        axs[1,1].plot(th_arr,precision_val[i])
        axs[2,1].plot(th_arr,recall_val[i])
    axs[0,1].plot(th_arr,np.nanmean(F1_val,axis=0),'k.-')
    axs[0,1].set_title('Validation F1')
    axs[0,1].set_ylabel('F1')
    axs[0,1].set_xlabel('Threshold')
	
    axs[1,1].plot(th_arr,np.nanmean(precision_val,axis=0),'k.-')
    axs[1,1].set_title('Validation precision')
    axs[1,1].set_ylabel('precision')
    axs[1,1].set_xlabel('Threshold')
	
    axs[2,1].plot(th_arr,np.nanmean(recall_val,axis=0),'k.-')
    axs[2,1].set_title('Validation recall')
    axs[2,1].set_ylabel('recall')
    axs[2,1].set_xlabel('Threshold')
  
    plt.close('all')

    plt.show()
    fig.savefig(save_path + '/retrain_CNN_performance.png')

