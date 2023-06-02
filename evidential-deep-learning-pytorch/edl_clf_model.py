import torch
import torch.nn as nn


class DenseDirichletLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dense = nn.Linear(in_feat, out_feat)

    def forward(self, x):
        # print(f'Shape of X - {x.shape}')
        output = self.dense(x)
        # print(f'Shape pf output - {output.shape}')
        evidence = torch.exp(output)
        # print(f'Shape of Evidence - {evidence.shape}')
        alpha = evidence + 1
        # print(f'Shape of alpha - {alpha.shape}')
        prob = alpha / torch.sum(alpha, axis=1, keepdim=True)
        # print(f'Shape of prob - {prob.shape}')
        return prob, alpha
    

class edl_clf_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, 100)
        self.l2 = nn.Linear(100, 50)
        self.dirichlet_layer = DenseDirichletLayer(50, 10)
        self.beta = nn.Parameter(torch.FloatTensor(torch.ones((1, 10))))
        print(f'Model Instantiated \n{self}')

    def forward(self, x):
        x = self.l1(x)
        X = nn.ReLU()(x)
        x = self.l2(x)
        x = nn.Tanh()(x)
        prob, alpha = self.dirichlet_layer(x)
        return prob, alpha
    
    def loss(self, alpha, label):

        def KL(alpha):
            beta = self.beta
            S_alpha = torch.sum(alpha, axis = 1, keepdim=True)
            S_beta = torch.sum(beta, axis = 1, keepdim=True)
            lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), axis = 1, keepdim=True)
            lnB_uni = torch.sum(torch.lgamma(beta), axis=1, keepdim=True) - torch.lgamma(S_beta)
            
            dg0  = torch.digamma(S_alpha)
            dg1 = torch.digamma(alpha)

            kl = torch.sum((alpha - beta)*(dg1-dg0), axis = 1, keepdim=True) + lnB + lnB_uni
            return kl

        S = torch.sum(alpha, axis=1, keepdim=True)
        m = alpha/S
        
        A = torch.sum((label - m)**2, axis=1, keepdim=True)
        B = torch.sum(alpha*(S - alpha)/(S*S*(S+1)), axis=1, keepdim=True)
        
        alpha_hat = label + (1-label)*alpha
        C = KL(alpha_hat)
        C = torch.mean(C, axis = 1)
        return torch.mean(A + B + C)
    
    