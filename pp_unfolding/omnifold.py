import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from t
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

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
    
    w_data = np.ones(len(theta_unknown_S[theta_unknown_S!=dummyval]))
    
    weights = np.empty(shape=(iterations, 2, len(theta0)))
    # shape = (iteration, step, event)
    
    theta0_G = theta0[:,0]
    theta0_S = theta0[:,1]
    
    labels0 = np.zeros(len(theta0))
    labels_unknown = np.ones(len(theta_unknown_S))
    
    xvals_1 = np.concatenate((theta0_S, theta_unknown_S[theta_unknown_S!=dummyval]))
    yvals_1 = np.concatenate((labels0, np.ones(len(theta_unknown_S[theta_unknown_S!=dummyval]))))

    xvals_2 = np.concatenate((theta0_G, theta0_G))
    yvals_2 = np.concatenate((labels0, labels_unknown))

    # initial iterative weights are ones
    weights_pull = np.ones(len(theta0_S))
    weights_push = np.ones(len(theta0_S))
    
    for i in range(iterations):
        print("\nITERATION: {}\n".format(i + 1))

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data
        print("STEP 1\n")

        weights_1 = np.concatenate((weights_push, w_data))
        #QUESTION: concatenation here confuses me
        # actual weights for Sim., ones for Data (not MC weights)

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(
            xvals_1, yvals_1, weights_1) #REMINDER: made up of synthetic+measured

        model.compile(loss='binary_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'])
        model.fit(X_train_1[X_train_1!=dummyval],
                Y_train_1[X_train_1!=dummyval],
                sample_weight=w_train_1[X_train_1!=dummyval],
                epochs=200,
                batch_size=10000,
                validation_data=(X_test_1[X_test_1!=dummyval], Y_test_1[X_test_1!=dummyval], w_test_1[X_test_1!=dummyval]),
                callbacks=[earlystopping],
                verbose=1)

        weights_pull = weights_push * reweight(theta0_S) 

        # STEP 1B: Need to do something with events that don't pass reco.
        
        #One option is to take the prior:
        #weights_pull[theta0_S==dummyval] = 1. 
        
        #Another option is to assign the average weight: <w|x_true>.  To do this, we need to estimate this quantity.
        xvals_1b = np.concatenate([theta0_G[theta0_S!=dummyval],theta0_G[theta0_S!=dummyval]])
        yvals_1b = np.concatenate([np.ones(len(theta0_G[theta0_S!=dummyval])),np.zeros(len(theta0_G[theta0_S!=dummyval]))])
        weights_1b = np.concatenate([weights_pull[theta0_S!=dummyval],np.ones(len(theta0_G[theta0_S!=dummyval]))])
        
        X_train_1b, X_test_1b, Y_train_1b, Y_test_1b, w_train_1b, w_test_1b = train_test_split(
            xvals_1b, yvals_1b, weights_1b)    
        
        model.compile(loss='binary_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'])
        model.fit(X_train_1b,
                Y_train_1b,
                sample_weight=w_train_1b,
                epochs=200,
                batch_size=10000,
                validation_data=(X_test_1b, Y_test_1b, w_test_1b),
                callbacks=[earlystopping],
                verbose=1)
        
        average_vals = reweight(theta0_G[theta0_S==dummyval])
        weights_pull[theta0_S==dummyval] = average_vals
        
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
                    metrics=['accuracy'])
        model.fit(X_train_2,
                Y_train_2,
                sample_weight=w_train_2,
                epochs=200,
                batch_size=2000,
                validation_data=(X_test_2, Y_test_2, w_test_2),
                callbacks=[earlystopping],
                verbose=1)

        weights_push = reweight(theta0_G)
        
        # STEP 2B: Need to do something with events that don't pass truth    
        
        #One option is to take the prior:
        #weights_push[theta0_G==dummyval] = 1. 
        
        #Another option is to assign the average weight: <w|x_reco>.  To do this, we need to estimate this quantity.
        xvals_1b = np.concatenate([theta0_S[theta0_G!=dummyval],theta0_S[theta0_G!=dummyval]])
        yvals_1b = np.concatenate([np.ones(len(theta0_S[theta0_G!=dummyval])),np.zeros(len(theta0_S[theta0_G!=dummyval]))])
        weights_1b = np.concatenate([weights_push[theta0_G!=dummyval],np.ones(len(theta0_S[theta0_G!=dummyval]))])
        
        X_train_1b, X_test_1b, Y_train_1b, Y_test_1b, w_train_1b, w_test_1b = train_test_split(
            xvals_1b, yvals_1b, weights_1b)    
        
        model.compile(loss='binary_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'])
        model.fit(X_train_1b,
                Y_train_1b,
                sample_weight=w_train_1b,
                epochs=200,
                batch_size=10000,
                validation_data=(X_test_1b, Y_test_1b, w_test_1b),
                callbacks=[earlystopping],
                verbose=1)
        
        average_vals = reweight(theta0_S[theta0_G==dummyval])
        weights_push[theta0_G==dummyval] = average_vals  
        
        weights[i, 1:2, :] = weights_push
        
    return weights