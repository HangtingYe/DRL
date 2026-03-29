import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import ipdb

def random_orthogonal_vectors(num_vectors, vector_dim):
    # ensure linear independent
    while True:
        random_matrix = np.random.randn(num_vectors, vector_dim)
        if np.linalg.matrix_rank(random_matrix) == num_vectors:
            break
    
    # initialize
    orthogonal_vectors = np.zeros((num_vectors, vector_dim))
    
    # Gram-Schmidt process
    for i in range(num_vectors):
        v = random_matrix[i]
        
        for j in range(i):
            v -= np.dot(v, orthogonal_vectors[j]) * orthogonal_vectors[j]
        
        # normalize current vector
        orthogonal_vectors[i] = v / np.linalg.norm(v)
    
    return orthogonal_vectors

class PN(nn.Module):
    def __init__(self, model_config):
        super(PN, self).__init__()
        self.data_dim = model_config['data_dim']
        self.hidden_dim = model_config['hidden_dim']

        self.en_nlayers = model_config['en_nlayers']
        self.de_nlayers = model_config['de_nlayers']
        self.model_config = model_config
        
        if model_config['diversity'] == False:
            if model_config['plearn'] == False:
                self.prototype = nn.Parameter(torch.rand(model_config['prototype_num'], self.hidden_dim), requires_grad=False)
            else:
                self.prototype = nn.Parameter(torch.rand(model_config['prototype_num'], self.hidden_dim), requires_grad=True)
        else:
            if model_config['plearn'] == False:
                self.prototype = nn.Parameter(torch.tensor(random_orthogonal_vectors(model_config['prototype_num'], self.hidden_dim)).float(), requires_grad=False)
            else:
                self.prototype = nn.Parameter(torch.tensor(random_orthogonal_vectors(model_config['prototype_num'], self.hidden_dim)).float(), requires_grad=True)

        phi = []
        encoder_dim = self.data_dim
        for _ in range(self.en_nlayers-2):
            phi.append(nn.Linear(encoder_dim,self.hidden_dim,bias=False))
            phi.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim
        phi.append(nn.Linear(encoder_dim,model_config['prototype_num'],bias=False))
        self.phi = nn.Sequential(*phi)

        encoder = []
        encoder_dim = self.data_dim
        for _ in range(self.en_nlayers-1):
            encoder.append(nn.Linear(encoder_dim,self.hidden_dim,bias=False))
            encoder.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        # for _ in range(self.de_nlayers-1):
        #     decoder.append(nn.Linear(self.hidden_dim,self.hidden_dim,bias=False))
        #     decoder.append(nn.LeakyReLU(0.2, inplace=True))
        decoder.append(nn.Linear(self.hidden_dim,self.data_dim,bias=False))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x_input):
        h = self.encoder(x_input)

        weight = self.phi(x_input)
        # !!!!!!!!!
        # weight = F.softmax(self.phi(x_input), dim=1)
        h_ = weight@self.prototype

        mse = F.mse_loss(h, h_, reduction='none')

        l2_norm_square = mse.sum(dim=1,keepdim=True)
        l2_norm_square_normalize = mse.sum(dim=1,keepdim=True) / torch.sum(h**2,dim=1,keepdim=True)
        l2_norm_square = torch.cat([l2_norm_square,l2_norm_square_normalize],dim=1)
        l2_norm_square,_ = torch.max(l2_norm_square, dim=1, keepdim=True)

        return l2_norm_square

    def predict_score(self, x_input):
        h = self.encoder(x_input)

        weight = self.phi(x_input)
        # !!!!!!!!!
        # weight = F.softmax(self.phi(x_input), dim=1)
        h_ = weight@self.prototype

        mse = F.mse_loss(h, h_, reduction='none')

        l2_norm_square = mse.sum(dim=1,keepdim=True)

        return l2_norm_square
    
