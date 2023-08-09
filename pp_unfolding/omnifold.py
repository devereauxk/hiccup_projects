import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras.models import Model
import pickle
import yaml, json
import horovod.tensorflow.keras as hvd
from dense import MLP

dummyval = -9999

# Binary crossentropy for classifying two samples with weights
# Weights are "hidden" by zipping in y_true (the labels)
def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(t_loss)

def save_object(obj, dir):
    """
        obj: the object (pointer)
        dir: the full directory wrt this utils.py file including the file name it should be stored in and the file type
            (ex: "dir/file_name.p")
        saves the object in the directory file specified
    """
    file = open(dir, "wb")
    pickle.dump(obj, file)
    file.close()
    
def load_object(dir):
    """
        dir: the full directory wrt this utils.py file including the file name it should be stored in and the file type
            (ex: "dir/file_name.p")
        returns the object saved in this directory
    """
    file = open(dir, "rb")
    obj = pickle.load(file)
    file.close()
    return obj

def LoadJson(file_name):
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

class Multifold():
    def __init__(self,nevts,mc_gen=None,mc_reco=None,data=None,config_file='config_omnifold.json',verbose=1):

        self.nevts = nevts
        self.verbose = verbose

        self.opt = LoadJson(config_file)
        self.niter = self.opt['General']['NITER']
        self.BATCH_SIZE= self.opt['General']['BATCH_SIZE']
        self.EPOCHS= self.opt['General']['EPOCHS']
        self.lr = float(self.opt['General']['LR'])

        self.mc_gen = mc_gen
        self.mc_reco = mc_reco
        self.data = data

        self.weights = weights = np.empty(shape=(self.niter, 2, self.mc_gen.shape[0]))
        # shape = (iteration, step, event)

        self.weights_folder = './weights'
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)
            
    def Unfold(self):
        self.CompileModel()
        
        for i in range(self.niter):
            print("ITERATION: {}".format(i + 1))            
            self.RunStep1(i)        
            self.RunStep2(i)            
                

    def RunStep1(self,i):
        '''Data versus reco MC reweighting'''
        print("RUNNING STEP 1")
        
        """
        self.RunModel(
            np.concatenate((self.mc_reco, self.data[self.data[:,0]!=dummyval])),
            np.concatenate((np.zeros(len(self.mc_reco)), np.ones(len(self.data[self.data[:,0]!=dummyval])))),
            np.concatenate((self.weights_push, np.ones(len(self.data[self.data[:,0]!=dummyval])))),
            i,self.model1,stepn=1
        )
        
        self.weights_pull = self.weights_push * self.reweight(self.mc_reco,self.model1)
        
        """
        xvals_1 = np.concatenate((self.mc_reco, self.data[self.data[:,0]!=dummyval]))
        yvals_1 = np.concatenate((np.zeros(len(self.mc_reco)), np.ones(len(self.data[self.data[:,0]!=dummyval]))))
        weights_1 = np.concatenate((self.weights_push, np.ones(len(self.data[self.data[:,0]!=dummyval]))))
        
        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(
            xvals_1, yvals_1, weights_1) #REMINDER: made up of synthetic+measured
        
        verbose = 2 if hvd.rank()==0 else 0
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            ReduceLROnPlateau(patience=8, min_lr=1e-7,verbose=verbose),
            EarlyStopping(patience=self.opt['General']['NPATIENCE'],restore_best_weights=True)
        ]
        
        self.model.fit(X_train_1[X_train_1[:,0]!=dummyval],
                Y_train_1[X_train_1[:,0]!=dummyval],
                sample_weight=pd.Series(w_train_1[X_train_1[:,0]!=dummyval]),
                epochs=20,
                batch_size=2000,
                validation_data=(X_test_1[X_test_1[:,0]!=dummyval], Y_test_1[X_test_1[:,0]!=dummyval], w_test_1[X_test_1[:,0]!=dummyval]),
                callbacks=callbacks,
                verbose=verbose)
                
        self.weights_pull = self.weights_push * self.reweight(self.mc_reco,self.model)


        # STEP 1B: Need to do something with events that don't pass reco.
        # one option is to simply do:
        # new_weights[self.not_pass_reco]=1.0
        # Another option is to assign the average weight: <w|x_true>.  To do this, we need to estimate this quantity.
        print("RUNNING STEP 1B")
        
        """
        self.RunModel(
            np.concatenate((self.mc_gen[self.pass_reco], self.mc_gen[self.pass_reco])),
            np.concatenate((np.ones(len(self.mc_gen[self.pass_reco])), np.zeros(len(self.mc_gen[self.pass_reco])))),
            np.concatenate((self.weights_pull[self.pass_reco], np.ones(len(self.mc_gen[self.pass_reco])))),
            i,self.model1b,stepn=1.5
        )
        
        average_vals = self.reweight(self.mc_gen[self.not_pass_reco], self.model1b)
        self.weights_pull[self.not_pass_reco] = average_vals
        
        """
        xvals_1b = np.concatenate((self.mc_gen[self.pass_reco], self.mc_gen[self.pass_reco]))
        yvals_1b = np.concatenate((np.ones(len(self.mc_gen[self.pass_reco])), np.zeros(len(self.mc_gen[self.pass_reco]))))
        weights_1b = np.concatenate((self.weights_pull[self.pass_reco], np.ones(len(self.mc_gen[self.pass_reco]))))
        
        X_train_1b, X_test_1b, Y_train_1b, Y_test_1b, w_train_1b, w_test_1b = train_test_split(
            xvals_1b, yvals_1b, weights_1b)
        
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            ReduceLROnPlateau(patience=8, min_lr=1e-7,verbose=verbose),
            EarlyStopping(patience=self.opt['General']['NPATIENCE'],restore_best_weights=True)
        ]
        self.model.fit(X_train_1b,
                Y_train_1b,
                sample_weight=w_train_1b,
                epochs=20,
                batch_size=10000,
                validation_data=(X_test_1b, Y_test_1b, w_test_1b),
                callbacks=callbacks,
                verbose=verbose)
                
        average_vals = self.reweight(self.mc_gen[self.not_pass_reco], self.model)
        self.weights_pull[self.not_pass_reco] = average_vals #TODO confirm this line works as intended
        

        # end of STEP 1B
        
        self.weights[i, :1, :] = self.weights_pull

    def RunStep2(self,i):
        '''Gen to Gen reweighing'''        
        print("RUNNING STEP 2")
        
        """
        self.RunModel(
            np.concatenate((self.mc_gen, self.mc_gen)),
            np.concatenate((np.zeros(len(self.mc_gen)), np.ones(len(self.mc_gen)))),
            np.concatenate((np.ones(len(self.mc_gen)), self.weights_pull)),
            i,self.model2,stepn=2
        )
        
        new_weights=self.reweight(self.mc_gen,self.model2)
        
        """
        xvals_2 = np.concatenate((self.mc_gen, self.mc_gen))
        yvals_2 = np.concatenate((np.zeros(len(self.mc_gen)), np.ones(len(self.mc_gen))))
        weights_2 = np.concatenate((np.ones(len(self.mc_gen)), self.weights_pull))
        
        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(
            xvals_2, yvals_2, weights_2)
        
        verbose = 2 if hvd.rank()==0 else 0
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            ReduceLROnPlateau(patience=8, min_lr=1e-7,verbose=verbose),
            EarlyStopping(patience=self.opt['General']['NPATIENCE'],restore_best_weights=True)
        ]
        self.model.fit(X_train_2,
                Y_train_2,
                sample_weight=w_train_2,
                epochs=20,
                batch_size=2000,
                validation_data=(X_test_2, Y_test_2, w_test_2),
                callbacks=callbacks,
                verbose=verbose)

        new_weights=self.reweight(self.mc_gen,self.model)
        
        
        new_weights[self.not_pass_gen]=1.0
        self.weights_push = new_weights
        self.weights[i, 1:2, :] = self.weights_push

    def RunModel(self,sample,labels,weights,iteration,model,stepn):
        
        # remove automatically events that don't pass reco or truth
        """
        mask = sample[:,0]!=dummyval
        if self.verbose: print("SHUFFLE BUFFER",np.sum(mask))
        data = tf.data.Dataset.from_tensor_slices((
            sample[mask],
            np.stack((labels[mask],weights[mask]),axis=1))
        ).cache().shuffle(np.sum(mask))
        """
        
        # IMPORTANT: assume sample,labels,weights already deals with events that dont pass reco/gen
        shuffle_size = sample.shape[0] # if slow, change to shuffle_size to a number larger than self.BATCH_SIZE>
        data = tf.data.Dataset.from_tensor_slices((
            sample,
            np.stack((labels,weights),axis=1))
        ).cache().shuffle(shuffle_size) 

        #Fix same number of training events between ranks
        NTRAIN,NTEST = self.GetNtrainNtest(stepn)        
        test_data = data.take(NTEST).repeat().batch(self.BATCH_SIZE)
        train_data = data.skip(NTEST).repeat().batch(self.BATCH_SIZE)

        if self.verbose:
            print(80*'#')
            print("Train events used: {}, total number of train events: {}, percentage: {}".format(NTRAIN,int(np.sum(shuffle_size)*0.8), NTRAIN/(np.sum(shuffle_size)*0.8)))
            print(80*'#')

        verbose = 2 if hvd.rank() == 0 else 0
        
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            # hvd.callbacks.LearningRateWarmupCallback(
            #     initial_lr=self.hvd_lr, warmup_epochs=self.opt['General']['NWARMUP'],
            #     verbose=verbose),
            ReduceLROnPlateau(patience=8, min_lr=1e-7,verbose=verbose),
            EarlyStopping(patience=self.opt['General']['NPATIENCE'],restore_best_weights=True)
        ]
        
        base_name = "Omnifold"
        
        if hvd.rank() ==0:
            callbacks.append(
                ModelCheckpoint('{}/{}_iter{}_step{}.h5'.format(self.weights_folder,base_name,iteration,stepn),
                                save_best_only=True,mode='auto',period=1,save_weights_only=True))
            
        _ =  model.fit(
            train_data,
            epochs=self.EPOCHS,
            steps_per_epoch=int(NTRAIN/self.BATCH_SIZE),
            validation_data=test_data,
            validation_steps=int(NTEST/self.BATCH_SIZE),
            verbose=verbose,
            callbacks=callbacks)


    def Preprocessing(self,weights_mc=None,weights_data=None):
        self.PrepareWeights(weights_mc,weights_data)
        self.PrepareModel()

    def PrepareWeights(self,weights_mc,weights_data):
        self.pass_reco = self.mc_reco[:,0]!=dummyval
        self.not_pass_reco = self.mc_reco[:,0]==dummyval
        self.not_pass_gen = self.mc_gen[:,0]==dummyval
        
        if weights_mc is None:
            self.weights_mc = np.ones(self.mc_reco.shape[0])
        else:
            self.weights_mc = weights_mc

        if weights_data is None:
            self.weights_data = np.ones(self.data[self.data[:,0]!=dummyval].shape[0])
        else:
            self.weights_data =weights_data
            
        self.weights_pull = np.ones(len(self.weights_mc))
        self.weights_push = np.ones(len(self.weights_mc))

    def PrepareModel(self):
        nvars = self.mc_gen.shape[1]
        inputs1,outputs1 = MLP(nvars,self.opt['MLP']['NTRIAL'])
        inputs1b,outputs1b = MLP(nvars,self.opt['MLP']['NTRIAL'])
        inputs2,outputs2 = MLP(nvars,self.opt['MLP']['NTRIAL'])
            
        self.model1 = Model(inputs=inputs1, outputs=outputs1)
        self.model1b = Model(inputs=inputs1b, outputs=outputs1b)
        self.model2 = Model(inputs=inputs2, outputs=outputs2)
        
        inputs = Input((nvars, ))
        hidden_layer_1 = Dense(50, activation='relu')(inputs)
        hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
        hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
        outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
        self.model = Model(inputs=inputs, outputs=outputs)

    def CompileModel(self):
        self.hvd_lr = self.lr
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=self.hvd_lr)
        opt = hvd.DistributedOptimizer(
            opt, average_aggregated_gradients=True)

        self.model1.compile(loss=weighted_binary_crossentropy,
                            optimizer=opt,experimental_run_tf_function=False)
        
        self.model1b.compile(loss=weighted_binary_crossentropy,
                            optimizer=opt,experimental_run_tf_function=False)

        self.model2.compile(loss=weighted_binary_crossentropy,
                            optimizer=opt,experimental_run_tf_function=False)

        self.model.compile(loss='binary_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'],
                           weighted_metrics=[])
        

    def GetNtrainNtest(self,stepn):
        if stepn ==1:
            # TODO optimize this
            # use only ~15% of data for step=1 (reco events)
            NTRAIN=int(0.2*0.8*self.nevts/hvd.size())
            NTEST=int(0.2*0.2*self.nevts/hvd.size())                        
        else:
            NTRAIN=int(0.8*self.nevts/hvd.size())
            NTEST=int(0.2*self.nevts/hvd.size())                        

        return NTRAIN,NTEST

    def reweight(self,events,model):
        f = np.nan_to_num(model.predict(events, batch_size=10000),posinf=1,neginf=0)
        weights = f / (1. - f)
        #weights = np.clip(weights,0,10)
        weights = weights[:,0]
        return np.squeeze(np.nan_to_num(weights,posinf=1))
    
    def GetWeights(self):
        return self.weights
    
