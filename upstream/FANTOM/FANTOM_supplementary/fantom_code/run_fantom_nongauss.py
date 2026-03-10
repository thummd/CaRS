import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import torch
import pyro.distributions as distrib
import torch.distributions as td
import torch.nn as nn
import fantom
from metrics import shd, classification_metrics, threshold_metrics, define_big_mat
from sklearn.metrics import accuracy_score,roc_auc_score
from torch.utils.data import DataLoader
from generate_nonlin_diff_var import gen_stationary_dyn_net_and_df_regime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
import yaml
#from fantom_train import FANTOM


device = "cuda:0" if torch.cuda.is_available() else "cpu"

with open("../fantom_code/config_gen_nongauss.yaml", "r") as f:
    config_gen = yaml.safe_load(f)

func_map = {
    "tanh": np.tanh,
    "exp": np.exp,
    "relu": lambda x: np.maximum(0, x),  # example
    # add more if needed
}
config_gen["funcs"] = [func_map[name] for name in config_gen["funcs"]]

g_list, df_total, intra_nodes, inter_nodes = gen_stationary_dyn_net_and_df_regime(
                                                                                          config_gen["n_regimes"],
                                                                                          config_gen["funcs"],
                                                                                          config_gen["n_nodes"],
                                                                                          config_gen["reg"],
                                                                                          config_gen["lag"],
                                                                                          config_gen["graph_ord_lag"],
                                                                                          config_gen["graph_ord_inst"],
                                                                                          w_max_inter = config_gen["w_max_inter"],
                                                                                          w_min_inter = config_gen["w_min_inter"],
                                                                                          w_max_intra = config_gen["w_max_intra"],
                                                                                          w_min_intra = config_gen["w_min_intra"],
                                                                                          graph_type_intra = config_gen["graph_type_intra"],
                                                                                          graph_type_inter = config_gen["graph_type_inter"],
                                                                                          noise_scale = config_gen["noise_scale"],
                                                                                          max_data_gen_trials = config_gen["max_data_gen_trials"],
                                                                                          sem_type = config_gen["sem_type"]
                                                                                          )
# rearrage data after generating it
lag = False
if lag == True:
            rearange_intra = [str(i)+"_lag0" for i in range(config_gen["n_nodes"])]
            rearange_inter1 = [str(i)+"_lag1" for i in range(config_gen["n_nodes"])]
            rearange_inter2 = [str(i)+"_lag2" for i in range(config_gen["n_nodes"])]
            rearange_inter = rearange_inter1 + rearange_inter2
else:
            rearange_intra = [str(i)+"_lag0" for i in range(config_gen["n_nodes"])]
            rearange_inter = [str(i)+"_lag1" for i in range(config_gen["n_nodes"])]
            
data = df_total[rearange_inter+rearange_intra].to_numpy()
data = data.reshape((np.array(config_gen["reg"]).sum(),2,config_gen["n_nodes"]))

#config model
with open("../fantom_code/config_nongauss_10.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_config = config['dataset_config']
model_config = config['model_config']
training_params = config['training_params']
#Prior
class pi_tn(nn.Module):
            def __init__(self, regime):
                        super(pi_tn, self).__init__()
                        self.regime = regime
                        self.linear = nn.Linear(1,self.regime)


            def forward(self,t):
                        outp = self.linear(t)
                        return outp
         
def train(num_epochs, model, data, gamma):
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            soft = nn.Softmax()
            for _ in range(num_epochs):
                        y_pre = model(data)
                        loss = criterion(y_pre, gamma)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            #print("loss: "+str(loss.item()))


            return soft(y_pre),loss.item(),model
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


gamma_hat, model_n = FANTOM(data, 3, config_gen["n_nodes"], 1, 1500, 1200, 0.85, device,"non-Gauss")
n_reg = config_gen["n_regimes"]
adj_mat = []
node = config_gen["n_nodes"]
for i in range(n_reg):
            adj_mat.append(model_n[i].get_adj_matrix(
                                    samples=1, most_likely_graph=True, squeeze=True
                                     ))
all_estimated_g,all_estimated_g_lag = define_big_mat(node,n_reg, adj_mat,g_list,False)
g_3_regime,g_3_regime_lag = define_big_mat(node,n_reg, adj_mat, g_list,True)

print("SHD for Inst: "+str(shd(g_3_regime,all_estimated_g))+", F1 for Inst: "+str( classification_metrics(g_3_regime,all_estimated_g.astype(int))['f1'])+
      ", SHD for lags: "+str(shd(g_3_regime_lag,all_estimated_g_lag))+", F1 for lags: "+str(classification_metrics(g_3_regime_lag,all_estimated_g_lag.astype(int))['f1'])
      )