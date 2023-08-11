import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Flatten
from keras.models import Model
import pickle
import yaml, json
import horovod.tensorflow.keras as hvd
from dense import MLP

dummyval = -1

def split_by_index(arr, index):
    # arr is array with EEC pair information in each row
    # the output is arr split into different arrays
    # each one for a unique occuarnce in the specifiecd index column
    output = []
    this_jet = []
    this_jet_pt = arr[0][index]
    for row in arr:
        if row[index] == this_jet_pt:
            this_jet.append(row)
        else:
            output.append(np.array(this_jet))
            this_jet = [row]
            this_jet_pt = row[index]
    return np.array(output, dtype=object)

def pad_out_splits(arr,dummyval=-10,pad_length=511):
    # subarrays with length < pad_legnth are padded with dummyval
    # subarrays with length > pad_length are removed from the set
    output = []
    for subset in arr:
        if subset.shape[0] > pad_length:
            continue
        
        dummy_part = dummyval * np.ones((pad_length - subset.shape[0], subset.shape[1]))
        
        subset = np.vstack([subset, dummy_part])
        output.append(subset)
        
    return np.array(output)
    
        
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
        
        self.RunModel_New(
            np.concatenate((self.mc_reco[self.mc_pass_reco], self.data[self.data_pass_reco])),
            np.concatenate((np.zeros(len(self.mc_reco[self.mc_pass_reco])), np.ones(len(self.data[self.data_pass_reco])))),
            np.concatenate((self.weights_push[self.mc_pass_reco], np.ones(len(self.data[self.data_pass_reco])))),
            i, self.model1, 1
        )
                
        self.weights_pull = self.weights_push * self.reweight(self.mc_reco,self.model1)
        
        """
        self.RunModel_Old(
            np.concatenate((self.mc_reco[self.mc_pass_reco], self.data[self.data_pass_reco])),
            np.concatenate((np.zeros(len(self.mc_reco[self.mc_pass_reco])), np.ones(len(self.data[self.data_pass_reco])))),
            np.concatenate((self.weights_push[self.mc_pass_reco], np.ones(len(self.data[self.data_pass_reco]))))
        )
                
        self.weights_pull = self.weights_push * self.reweight(self.mc_reco,self.model)
        """
        

        # STEP 1B: Need to do something with events that don't pass reco.
        # one option is to simply do:
        # new_weights[self.not_pass_reco]=1.0
        # Another option is to assign the average weight: <w|x_true>.  To do this, we need to estimate this quantity.
        print("RUNNING STEP 1B")
        
        self.RunModel_New(
            np.concatenate((self.mc_gen[self.mc_pass_reco], self.mc_gen[self.mc_pass_reco])),
            np.concatenate((np.ones(len(self.mc_gen[self.mc_pass_reco])), np.zeros(len(self.mc_gen[self.mc_pass_reco])))),
            np.concatenate((self.weights_pull[self.mc_pass_reco]*self.weights_mc[self.mc_pass_reco], np.ones(len(self.mc_gen[self.mc_pass_reco])))),
            i, self.model1b, 1.5
        )
        
        average_vals = self.reweight(self.mc_gen[self.not_mc_pass_reco], self.model1b)
        self.weights_pull[self.not_mc_pass_reco] = average_vals
        
        """
        self.RunModel_Old(
            np.concatenate((self.mc_gen[self.mc_pass_reco], self.mc_gen[self.mc_pass_reco])),
            np.concatenate((np.ones(len(self.mc_gen[self.mc_pass_reco])), np.zeros(len(self.mc_gen[self.mc_pass_reco])))),
            np.concatenate((self.weights_pull[self.mc_pass_reco], np.ones(len(self.mc_gen[self.mc_pass_reco]))))
        )
        
        average_vals = self.reweight(self.mc_gen[self.not_mc_pass_reco], self.model)
        self.weights_pull[self.not_mc_pass_reco] = average_vals
        """
        
        
        # end of STEP 1B
        
        self.weights[i, :1, :] = self.weights_pull

        
    def RunStep2(self,i):
        '''Gen to Gen reweighing'''        
        print("RUNNING STEP 2")
        
        
        self.RunModel_New(
            np.concatenate((self.mc_gen, self.mc_gen)),
            np.concatenate((np.zeros(len(self.mc_gen)), np.ones(len(self.mc_gen)))),
            np.concatenate((self.weights_mc, self.weights_mc*self.weights_pull)),
            i, self.model2, 2
        )
        
        new_weights=self.reweight(self.mc_gen,self.model2)
        """
        
        self.RunModel_Old(
            np.concatenate((self.mc_gen, self.mc_gen)),
            np.concatenate((np.zeros(len(self.mc_gen)), np.ones(len(self.mc_gen)))),
            np.concatenate((np.ones(len(self.mc_gen)), self.weights_pull))
        )
        
        new_weights=self.reweight(self.mc_gen,self.model)
        """
        
        new_weights[self.not_pass_gen]=1.0
        self.weights_push = new_weights
        self.weights[i, 1:2, :] = self.weights_push
        
        
        
    def RunModel_Old(self,xvals,yvals,weights):
        # assumes inputs contain the exact intended inputs for the model regardless of whether or not they pass
        # reco or not
        
        # IMPORTANT: assume sample,labels,weights already deals with events that dont pass reco/gen
        # if slow, change to shuffle_size to a number larger than self.BATCH_SIZE>
        shuffle_size = xvals.shape[0]
        data = tf.data.Dataset.from_tensor_slices((
            xvals,
            yvals,
            weights
        )).cache().shuffle(shuffle_size)
        
        #Fix same number of training events between ranks
        NTRAIN,NTEST = self.GetNtrainNtest(None)        
        test_data = data.take(NTEST).repeat().batch(self.BATCH_SIZE)
        train_data = data.skip(NTEST).repeat().batch(self.BATCH_SIZE)
        
        verbose = 2 if hvd.rank()==0 else 0
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            ReduceLROnPlateau(patience=8, min_lr=1e-7,verbose=verbose),
            EarlyStopping(patience=self.opt['General']['NPATIENCE'],restore_best_weights=True)
        ]
        """
        data_iterator = train_data.as_numpy_iterator()
        print(80*'#')
        print("DATA")
        x_batch, y_batch, weights_batch = next(data_iterator)
        print(f"Batch {0 + 1}: x_batch.shape = {x_batch.shape}, y_batch.shape = {y_batch.shape}, weights_batch.shape = {weights_batch.shape}")
        """
        
        self.model.fit(
                train_data,
                epochs=self.EPOCHS,
                steps_per_epoch=int(NTRAIN/self.BATCH_SIZE),
                validation_data=test_data,
                validation_steps=int(NTEST/self.BATCH_SIZE),
                callbacks=callbacks,
                verbose=verbose)
        

    def RunModel_New(self,sample,labels,weights,iteration,model,stepn):
        
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
        if len(self.mc_reco.shape) == 3: # input is 3D i.e. either jet or event level
            self.mc_pass_reco = self.mc_reco[:,0,0]!=dummyval
            self.not_mc_pass_reco = self.mc_reco[:,0,0]==dummyval
            self.data_pass_reco = self.data[:,0,0]!=dummyval
            self.not_pass_gen = self.mc_gen[:,0,0]==dummyval
        else: # assume input is 2D; i.e. pair level
            self.mc_pass_reco = self.mc_reco[:,0]!=dummyval
            self.not_mc_pass_reco = self.mc_reco[:,0]==dummyval
            self.data_pass_reco = self.data[:,0]!=dummyval
            self.not_pass_gen = self.mc_gen[:,0]==dummyval
        
        if weights_mc is None:
            self.weights_mc = np.ones(self.mc_reco.shape[0])
        else:
            self.weights_mc = weights_mc

        if weights_data is None:
            self.weights_data = np.ones(self.data[self.data_pass_reco].shape[0])
        else:
            self.weights_data =weights_data
            
        self.weights_pull = np.ones(len(self.weights_mc))
        self.weights_push = np.ones(len(self.weights_mc))

    def PrepareModel(self):
        input_shape = self.mc_gen.shape[1:]
        inputs1,outputs1 = MLP(input_shape,self.opt['MLP']['NTRIAL'])
        inputs1b,outputs1b = MLP(input_shape,self.opt['MLP']['NTRIAL'])
        inputs2,outputs2 = MLP(input_shape,self.opt['MLP']['NTRIAL'])
            
        self.model1 = Model(inputs=inputs1, outputs=outputs1)
        self.model1b = Model(inputs=inputs1b, outputs=outputs1b)
        self.model2 = Model(inputs=inputs2, outputs=outputs2)
        
        print("model input shape : " + str(self.mc_gen.shape[1:]))
        inputs = Input(self.mc_gen.shape[1:])
        hidden_layer_1 = Dense(50, activation='relu')(inputs)
        hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
        hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
        hidden_layer_flat = Flatten()(hidden_layer_2)
        outputs = Dense(1, activation='sigmoid')(hidden_layer_flat)
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
        
        # setting up the model like this ONLY works with the format of model.fit used in RunModel
        # i.e. not in the way used in the preliminary studies
        self.model.compile(loss=weighted_binary_crossentropy,
                            optimizer=opt,experimental_run_tf_function=False,
                            weighted_metrics=[])
        self.model.summary()
                            

    def GetNtrainNtest(self,stepn):
        # CHANGING THESE PARAMETERS CAN CHANGE FIT DRASTICALLY
        if stepn == 1:
            # TODO optimize this
            # use only ~15% of data for step=1 (reco events)
            NTRAIN=int(0.8*self.nevts/hvd.size())
            NTEST=int(0.2*self.nevts/hvd.size())                        
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
    
