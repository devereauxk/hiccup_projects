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

def reweight(events,model,batch_size=10000):
    f = model.predict(events, batch_size=batch_size)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights))

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

def omnifold(theta0,theta_unknown_S,iterations,model,verbose=0):

    weights = np.empty(shape=(iterations, 2, len(theta0)))
    # shape = (iteration, step, event)
    
    theta0_G = theta0[:,0]
    theta0_S = theta0[:,1]
    
    labels0 = np.zeros(len(theta0))
    labels_unknown = np.ones(len(theta_unknown_S))
    
    xvals_1 = np.concatenate((theta0_S, theta_unknown_S))
    yvals_1 = np.concatenate((labels0, labels_unknown))

    xvals_2 = np.concatenate((theta0_G, theta0_G))
    yvals_2 = np.concatenate((labels0, labels_unknown))

    # initial iterative weights are ones
    weights_pull = np.ones(len(theta0_S))
    weights_push = np.ones(len(theta0_S))
    
    for i in range(iterations):

        if (verbose>0):
            print("\nITERATION: {}\n".format(i + 1))
            pass
        
        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        if (verbose>0):
            print("STEP 1\n")
            pass
            
        weights_1 = np.concatenate((weights_push, np.ones(len(theta_unknown_S))))

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1, weights_1)

        # zip ("hide") the weights with the labels
        Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
        Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)   
        
        model.compile(loss=weighted_binary_crossentropy,
                      optimizer='Adam',
                      metrics=['accuracy'])
        
        model.fit(X_train_1,
                  Y_train_1,
                  epochs=20,
                  batch_size=10000,
                  validation_data=(X_test_1, Y_test_1),
                  verbose=verbose)

        weights_pull = weights_push * reweight(theta0_S,model)
        weights[i, :1, :] = weights_pull

        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        if (verbose>0):
            print("\nSTEP 2\n")
            pass

        weights_2 = np.concatenate((np.ones(len(theta0_G)), weights_pull))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2, weights_2)

        # zip ("hide") the weights with the labels
        Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
        Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)   
        
        model.compile(loss=weighted_binary_crossentropy,
                      optimizer='Adam',
                      metrics=['accuracy'])
        model.fit(X_train_2,
                  Y_train_2,
                  epochs=20,
                  batch_size=2000,
                  validation_data=(X_test_2, Y_test_2),
                  verbose=verbose)
        
        weights_push = reweight(theta0_G,model)
        weights[i, 1:2, :] = weights_push
        pass
        
    return weights

def omnifold_tr_eff(theta0,theta_unknown_S,iterations,model,dummyval=-9999):

    earlystopping = EarlyStopping(patience=10,
                              verbose=1,
                              restore_best_weights=True)
    
    w_data = np.ones(len(theta_unknown_S[theta_unknown_S[:,0]!=dummyval]))
    
    weights = np.empty(shape=(iterations, 2, len(theta0)))
    # shape = (iteration, step, event)
    
    theta0_G = theta0[:,0]
    theta0_S = theta0[:,1]
    
    xvals_1 = np.concatenate((theta0_S, theta_unknown_S[theta_unknown_S[:,0]!=dummyval]))
    yvals_1 = np.concatenate((np.zeros(len(theta0_S)), np.ones(len(theta_unknown_S[theta_unknown_S[:,0]!=dummyval]))))

    xvals_2 = np.concatenate((theta0_G, theta0_G))
    yvals_2 = np.concatenate((np.zeros(len(theta0_G)), np.ones(len(theta0_G))))

    # initial iterative weights are ones
    weights_pull = np.ones(len(theta0_S))
    weights_push = np.ones(len(theta0_S))
    
    print("tests")
    print(theta0_G.shape)
    print(theta0_S.shape)
    print(theta_unknown_S.shape)

    print(theta0_G[theta0_S[:,0]!=-9999].shape)
    print(theta0_G[theta0_S[:,0]==-9999].shape)
    
    
    for i in range(iterations):
        print("\nITERATION: {}\n".format(i + 1))

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data
        print("STEP 1\n")
        
        weights_1 = np.concatenate((weights_push, w_data))
        #QUESTION: concatenation here confuses me
        # actual weights for Sim., ones for Data (not MC weights)
        
        print("shufling weights")

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(
            xvals_1, yvals_1, weights_1) #REMINDER: made up of synthetic+measured
        
        print("compiling model")
        
        model.compile(loss='binary_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'],
                    weighted_metrics=[])
        
        print("fitting model")
        
        model.fit(X_train_1[X_train_1[:,0]!=dummyval],
                Y_train_1[X_train_1[:,0]!=dummyval],
                sample_weight=pd.Series(w_train_1[X_train_1[:,0]!=dummyval]),
                epochs=20,
                batch_size=2000,
                validation_data=(X_test_1[X_test_1[:,0]!=dummyval], Y_test_1[X_test_1[:,0]!=dummyval], w_test_1[X_test_1[:,0]!=dummyval]),
                callbacks=[earlystopping],
                verbose=1)

        weights_pull = weights_push * reweight(theta0_S, model) 

        # STEP 1B: Need to do something with events that don't pass reco.
        
        #One option is to take the prior:
        #weights_pull[theta0_S[:,0]==dummyval] = 1. 
        
        #Another option is to assign the average weight: <w|x_true>.  To do this, we need to estimate this quantity.
        xvals_1b = np.concatenate([theta0_G[theta0_S[:,0]!=dummyval],theta0_G[theta0_S[:,0]!=dummyval]])
        yvals_1b = np.concatenate([np.ones(len(theta0_G[theta0_S[:,0]!=dummyval])),np.zeros(len(theta0_G[theta0_S[:,0]!=dummyval]))])
        weights_1b = np.concatenate([weights_pull[theta0_S[:,0]!=dummyval],np.ones(len(theta0_G[theta0_S[:,0]!=dummyval]))])
        
        X_train_1b, X_test_1b, Y_train_1b, Y_test_1b, w_train_1b, w_test_1b = train_test_split(
            xvals_1b, yvals_1b, weights_1b)
        
        model.compile(loss='binary_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'],
                    weighted_metrics=[])
        model.fit(X_train_1b,
                Y_train_1b,
                sample_weight=w_train_1b,
                epochs=200,
                batch_size=10000,
                validation_data=(X_test_1b, Y_test_1b, w_test_1b),
                callbacks=[earlystopping],
                verbose=1)
        
        print(theta0_G[theta0_S[:,0]==dummyval])
        print(theta0_G[theta0_S[:,0]==dummyval].shape)
        
        average_vals = reweight(theta0_G[theta0_S[:,0]==dummyval], model)
        weights_pull[theta0_S[:,0]==dummyval] = average_vals
        
        weights[i, :1, :] = weights_pull

        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.
        print("\nSTEP 2\n")

        weights_2 = np.concatenate((np.ones(len(theta0_G)), weights_pull))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(
            xvals_2, yvals_2, weights_2)

        model.compile(loss='binary_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'],
                    weighted_metrics=[])
        model.fit(X_train_2,
                Y_train_2,
                sample_weight=w_train_2,
                epochs=200,
                batch_size=2000,
                validation_data=(X_test_2, Y_test_2, w_test_2),
                callbacks=[earlystopping],
                verbose=1)

        weights_push = reweight(theta0_G, model)
        
        # STEP 2B: Need to do something with events that don't pass truth    
        
        #One option is to take the prior:
        #weights_push[theta0_G[:,0]==dummyval] = 1. 
        
        #Another option is to assign the average weight: <w|x_reco>.  To do this, we need to estimate this quantity.
        """
        xvals_1b = np.concatenate([theta0_S[theta0_G[:,0]!=dummyval],theta0_S[theta0_G[:,0]!=dummyval]])
        yvals_1b = np.concatenate([np.ones(len(theta0_S[theta0_G[:,0]!=dummyval])),np.zeros(len(theta0_S[theta0_G[:,0]!=dummyval]))])
        weights_1b = np.concatenate([weights_push[theta0_G[:,0]!=dummyval],np.ones(len(theta0_S[theta0_G[:,0]!=dummyval]))])
        
        X_train_1b, X_test_1b, Y_train_1b, Y_test_1b, w_train_1b, w_test_1b = train_test_split(
            xvals_1b, yvals_1b, weights_1b)    
        
        model.compile(loss='binary_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'],
                    weighted_metrics=[])
        model.fit(X_train_1b,
                Y_train_1b,
                sample_weight=w_train_1b,
                epochs=200,
                batch_size=10000,
                validation_data=(X_test_1b, Y_test_1b, w_test_1b),
                callbacks=[earlystopping],
                verbose=1)
        
        average_vals = reweight(theta0_S[theta0_G[:,0]==dummyval], model)
        weights_push[theta0_G[:,0]==dummyval] = average_vals
        """

        
        weights[i, 1:2, :] = weights_push
        
    return weights


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
        # self.CompileModel()
        
        for i in range(self.niter):
            print("ITERATION: {}".format(i + 1))            
            self.RunStep1(i)        
            self.RunStep2(i)            
                

    def RunStep1(self,i):
        '''Data versus reco MC reweighting'''
        print("RUNNING STEP 1")
        
        """
        self.RunModel(
            np.concatenate((self.mc_reco[self.mc_reco[:,0]!=dummyval], self.data[self.data[:,0]!=dummyval])),
            np.concatenate((np.zeros(len(self.mc_reco[self.mc_reco[:,0]!=dummyval])), np.ones(len(self.data[self.data[:,0]!=dummyval])))),
            np.concatenate((self.weights_push[self.mc_reco[:,0]!=dummyval],self.weights_data)),
            i,self.model1,stepn=1
        )
        """
        
        xvals_1 = np.concatenate((self.mc_reco, self.data[self.data[:,0]!=dummyval]))
        yvals_1 = np.concatenate((np.zeros(len(self.mc_reco)), np.ones(len(self.data[self.data[:,0]!=dummyval]))))
        weights_1 = np.concatenate((self.weights_push, np.ones(len(self.data[self.data[:,0]!=dummyval]))))
        
        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(
            xvals_1, yvals_1, weights_1) #REMINDER: made up of synthetic+measured
        
        self.model.compile(loss='binary_crossentropy',
            optimizer='Adam',
            metrics=['accuracy'],
            weighted_metrics=[])
        
        self.model.fit(X_train_1[X_train_1[:,0]!=dummyval],
                Y_train_1[X_train_1[:,0]!=dummyval],
                sample_weight=pd.Series(w_train_1[X_train_1[:,0]!=dummyval]),
                epochs=20,
                batch_size=2000,
                validation_data=(X_test_1[X_test_1[:,0]!=dummyval], Y_test_1[X_test_1[:,0]!=dummyval], w_test_1[X_test_1[:,0]!=dummyval]),
                verbose=1)
        
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
        """
        
        xvals_1b = np.concatenate((self.mc_gen[self.pass_reco], self.mc_gen[self.pass_reco]))
        yvals_1b = np.concatenate((np.ones(len(self.mc_gen[self.pass_reco])), np.zeros(len(self.mc_gen[self.pass_reco]))))
        weights_1b = np.concatenate((self.weights_pull[self.pass_reco], np.ones(len(self.mc_gen[self.pass_reco]))))
        
        X_train_1b, X_test_1b, Y_train_1b, Y_test_1b, w_train_1b, w_test_1b = train_test_split(
            xvals_1b, yvals_1b, weights_1b)
        
        self.model.compile(loss='binary_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'],
                    weighted_metrics=[])
        self.model.fit(X_train_1b,
                Y_train_1b,
                sample_weight=w_train_1b,
                epochs=20,
                batch_size=10000,
                validation_data=(X_test_1b, Y_test_1b, w_test_1b),
                verbose=1)

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
        )"""
        
        xvals_2 = np.concatenate((self.mc_gen, self.mc_gen))
        yvals_2 = np.concatenate((np.zeros(len(self.mc_gen)), np.ones(len(self.mc_gen))))
        weights_2 = np.concatenate((np.ones(len(self.mc_gen)), self.weights_pull))
        
        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(
            xvals_2, yvals_2, weights_2)

        self.model.compile(loss='binary_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'],
                    weighted_metrics=[])
        self.model.fit(X_train_2,
                Y_train_2,
                sample_weight=w_train_2,
                epochs=20,
                batch_size=2000,
                validation_data=(X_test_2, Y_test_2, w_test_2),
                verbose=1)

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
        
        # assume sample,labels,weights already deals with events that dont pass reco
        data = tf.data.Dataset.from_tensor_slices((
            sample,
            np.stack((labels,weights),axis=1))
        ).cache().shuffle(sample.shape[0])

        #Fix same number of training events between ranks
        NTRAIN,NTEST = self.GetNtrainNtest(stepn)        
        test_data = data.take(NTEST).repeat().batch(self.BATCH_SIZE)
        train_data = data.skip(NTEST).repeat().batch(self.BATCH_SIZE)

        """
        if self.verbose:
            print(80*'#')
            print("Train events used: {}, total number of train events: {}, percentage: {}".format(NTRAIN,np.sum(mask)*0.8, np.sum(mask)*0.8/NTRAIN))
            print(80*'#')
        """

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
        self.PrepareInputs()
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

    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc_reco))
        self.labels_data = np.ones(len(self.data[self.data[:,0]!=dummyval]))
        self.labels_gen = np.ones(len(self.mc_gen))

    def PrepareModel(self):
        nvars = self.mc_gen.shape[1]
        """
        inputs1,outputs1 = MLP(nvars,self.opt['MLP']['NTRIAL'])
        inputs1b,outputs1b = MLP(nvars,self.opt['MLP']['NTRIAL'])
        inputs2,outputs2 = MLP(nvars,self.opt['MLP']['NTRIAL'])
            
        self.model1 = Model(inputs=inputs1, outputs=outputs1)
        self.model1b = Model(inputs=inputs1b, outputs=outputs1b)
        self.model2 = Model(inputs=inputs2, outputs=outputs2)
        """
        
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
        

    def GetNtrainNtest(self,stepn): #TODO understand this
        if stepn ==1:
            #about 20% acceptance for reco events
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
    