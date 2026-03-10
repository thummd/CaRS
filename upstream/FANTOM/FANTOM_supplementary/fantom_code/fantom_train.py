import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import torch
import fantom
import pyro.distributions as distrib
import torch.distributions as td
import torch.nn as nn
#from metrics import shd, classification_metrics, threshold_metrics
from sklearn.metrics import accuracy_score,roc_auc_score
from torch.utils.data import DataLoader




def FANTOM(X, max_it, n_nodes, lag, window, zeta, thresh, device, noise_type):
            n = int(X.shape[1]//2)
            m = X.shape[0]
            N_regime = m//window
            if noise_type == "hetero":
                X_std = np.zeros(X.shape)
                X_std[:,0,:] = (X[:,0,:] -X[:,0,:].mean(axis=0))/X[:,0,:].std(axis=0)
                X_std[:,1,:] = (X[:,1,:] -X[:,1,:].mean(axis=0))/X[:,1,:].std(axis=0)
            else:
                X_std = X
            for it in range(max_it):
                        if it >= 2:
                        #            model_config['lambda_sparse'] = 100
                                    training_params["encoder_layer_sizes"] = [32,32]
                                    training_params["decoder_layer_sizes"] = [32,32]
                                    training_params["max_auglag_inner_epochs"] = 3000
                                    if noise_type == "hetero": 
                                             training_params["max_steps_auglag"] = 2
                                    else:
                                             training_params["max_steps_auglag"] = 3               


                        model_n = [fantom.FANTOM_stationary(n_nodes, device, lag = lag, allow_instantaneous=True, **model_config) for _ in range(N_regime)]
                        device = torch.device('cuda:0')
                        yl = np.zeros((m, n , N_regime))
                        log_pdf_emission = np.zeros((m, N_regime))
                        if it == 0:
                                    p = np.zeros((m,N_regime))
                                    for c in range(N_regime):
                                                if c  == N_regime -1:
                                                            p[c*window:,c] = np.ones(m-c*window)
                                                            if noise_type =="hetero": 
                                                                initial_regime_data = (X[c*window:,:,:] - X[c*window:,:,:].mean(axis=0))/X[c*window:,:,:].std(axis=0)
                                                            else:
                                                                initial_regime_data = X[c*window:,:,:]
                                                            dataloader = DataLoader(initial_regime_data,training_params["batch_size"])
                                                            model_n[c].run_train(dataloader, window, training_params)
                                                            log_pdf_emission[:,c] = np.exp(model_n[c].log_prob(torch.tensor(X_std),1))

                                                else:
                                                            p[c*window:(c+1)*window,c] = np.ones(window)
                                                            if noise_type =="hetero": 
                                                                initial_regime_data = (X[c*window:(c+1)*window,:] - X[c*window:(c+1)*window,:].mean(axis=0))/X[c*window:(c+1)*window,:].std(axis=0)
                                                            else:
                                                                initial_regime_data = X[c*window:(c+1)*window,:]
                                                            dataloader = DataLoader(initial_regime_data,training_params["batch_size"])
                                                            model_n[c].run_train(dataloader,window, training_params)
                                                            log_pdf_emission[:,c] = np.exp(model_n[c].log_prob(torch.tensor(X_std),1))


                        else:
                                    for c in range(N_regime):
                                                gamma = gamma_hat[:,c]
                                                data = gamma.reshape((m,1,1))*X
                                                b = data[~np.all(data == 0, axis=2)]
                                                regime_data = b.reshape(b.shape[0]//(lag + 1), (lag + 1), n_nodes)
                                                if noise_type =="hetero":
                                                    for cte in range(lag+1):
                                                                  regime_data[:,cte,:] = (regime_data[:,cte,:] - regime_data[:,cte,:].mean(axis=0))/regime_data[:,cte,:].std(axis=0)
                                                else:
                                                    regime_data = regime_data
                                                dataloader = DataLoader(regime_data,training_params["batch_size"])
                                                model_n[c].run_train(dataloader,regime_data.shape[0], training_params)
                                                log_pdf_emission[:,c] = np.exp(model_n[c].log_prob(torch.tensor(X_std),1))
                        pall = 0
                        gamma_hat = np.zeros((m, N_regime))
                        for class_idx in range(N_regime):
                                    pall = pall + p[:,class_idx] * log_pdf_emission[:,class_idx]
                                    gamma_hat[:, class_idx] = p[:,class_idx] * log_pdf_emission[:,class_idx]
                        idx = np.argmax(gamma_hat/pall.reshape((m,1)), axis=-1)
                        gamma_hat = np.zeros( gamma_hat.shape )
                        gamma_hat[ np.arange(gamma_hat.shape[0]), idx] = 1

                        t = torch.tensor(np.linspace(0,20*N_regime,X.shape[0]).reshape((X.shape[0],1)))
                        model = pi_tn(N_regime)
                        p,loss,model_ = train(500,model, t.float(), torch.tensor(gamma_hat).float())
                        p = p.detach().numpy()
                        while loss>=thresh:
                              p,loss,model_ = train(100,model_, t.float(), torch.tensor(gamma_hat).float())
                              p = p.detach().numpy()

                        if it ==0:
                              pall = 0
                              gamma_hat = np.zeros((m, N_regime))
                              for class_idx in range(N_regime):

                                             pall = pall + p[:,class_idx] * log_pdf_emission[:,class_idx]
                                             gamma_hat[:, class_idx] = p[:,class_idx] * log_pdf_emission[:,class_idx]
                              idx = np.argmax(gamma_hat/pall.reshape((m,1)), axis=-1)
                              gamma_hat = np.zeros( gamma_hat.shape )
                              gamma_hat[ np.arange(gamma_hat.shape[0]), idx] = 1

                        gamma_sum = np.sum(gamma_hat, axis=0)
                        gamma_sum[gamma_sum<zeta] = 0
                        indexes, = np.where(gamma_sum!= 0)
                        gamma_hat = gamma_hat[:,indexes]
                        p = p[:,indexes]
                        N_regime = len(indexes)
                        print(str(np.sum(gamma_hat, axis=0))+" iter: "+str(it)+" , p: "+str(np.sum(p, axis=0)))
            return gamma_hat, model_n