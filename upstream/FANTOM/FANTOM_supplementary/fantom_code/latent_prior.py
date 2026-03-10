import torch
import torch.distributions as td
import torch.nn as nn


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