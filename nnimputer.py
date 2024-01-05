from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import chain
import pandas as pd
import numpy as np
from torch import nn, optim, mean, log, randint, rand_like, no_grad, compile, concat, where, lt, from_numpy, device, float32, int8
import tqdm

class GenNet(nn.Module):
   def __init__(self, num_features, *args, **kwargs_):
       super().__init__(*args, **kwargs_)
       self.hidden_dim = 512
       self.mlp = nn.Sequential(
           nn.Linear(num_features * 2, self.hidden_dim),
           nn.ReLU(),
           nn.Linear(self.hidden_dim, self.hidden_dim),
           nn.ReLU(),
           nn.Dropout1d(),
           nn.Linear(self.hidden_dim, num_features), 
           nn.Sigmoid()
       )
       
   def forward(self, data, mask):
       return self.mlp(concat([data, mask], dim=1))

class NNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, input_size, 
                 epochs=5, learning_rate=0.001, missing_prob=0.25, batch_size=128, verbose=False):
        self.input_size = input_size
        self.missing_prob = missing_prob
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device("cuda") 
        self.generator = GenNet(input_size).to(device=self.device, dtype=float32)
        self.discriminator =  GenNet(input_size).to(device=self.device, dtype=float32)
        self.learning_rate = learning_rate
        self.optim_gen = None
        self.optim_disc = None
        self.verbose = verbose
        self.mse_weight = 10.0
        self.hint_prob = 0.9
        self.reset()
        
    def reset(self):
        for layer in chain(self.generator.children(), self.discriminator.children()):
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.optim_gen = optim.Adam(self.generator.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        self.optim_disc = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        
    def disc_loss(self, with_noise, hint, present_mask):
        # with torch.no_grad():
        generated = self.generator(with_noise, present_mask)
        imputed = with_noise * present_mask + generated * (1-present_mask)
        
        disc_prob = self.discriminator(imputed, hint)
        loss = -mean(
            present_mask * log(disc_prob + 1e-7) + (1-present_mask) * log(1-disc_prob+1e-7)
        )
        return loss
    
    def gen_loss(self, with_noise, hint, present_mask):
        generated = self.generator(with_noise, present_mask)
        imputed = with_noise * present_mask + generated * (1-present_mask)
        
        disc_prob = self.discriminator(imputed, hint)
        loss1 = -mean(
            (1-present_mask) * log(disc_prob + 1e-7)
        )
        
        loss2 = mean((present_mask * with_noise - present_mask * generated)**2) / mean(present_mask.to(dtype=float32))
        # print(loss1, loss2)
        return loss1 + self.mse_weight * loss2
    
    @compile
    def fit(self, X: pd.DataFrame, y=None):
      """
      X: input to be imputed. Must be scaled to 0 to 1 range before  imputation.
         Including a scaler inside may be preferable, but it poses a problem during testing.
         When testing with unknown missing values in the test set, we want to impute the test set with 
         means from the training set, so that we know the missing values when testing. However, this mean 
         needs to be computed after scaling, and before imputation with this class.
      y: included for compatibility. Value ignored.
      """
        self.discriminator.train()
        self.generator.train()
        data = from_numpy(X.to_numpy())
        
        for i in tqdm.tqdm(range(self.epochs)):
            sample_ind = randint(0, len(data), size=(self.batch_size,))
            sample = data[sample_ind, :].to(device=self.device, dtype=float32)
            present_mask = sample.isnan().logical_not()
            noise = rand_like(sample) * 0.01
            hint_noise = lt(rand_like(sample), self.hint_prob)
            
            # The paper seems to indicate the 0.5 * (~hint_noise) is needed, but this is not included in the official implementation.
            # It also seems to give a worse result.
            hint = hint_noise * present_mask # + 0.5 * (~hint_noise)
            with_noise = where(present_mask, sample, noise)
            
            self.optim_disc.zero_grad()
            disc_loss = self.disc_loss(with_noise, hint.to(dtype=int8), present_mask.to(dtype=int8))
            disc_loss.backward()
            self.optim_disc.step()
            if self.verbose and i%100==0:
                print("{}: Disc loss={}".format(i, disc_loss))

            self.optim_gen.zero_grad()
            gen_loss = self.gen_loss(with_noise, hint.to(dtype=int8), present_mask.to(dtype=int8))
            gen_loss.backward()
            self.optim_gen.step()
            if self.verbose and i%100==0:
                print("{}: Gen loss={}".format(i, gen_loss))
        return self
    
    @compile
    def transform(self, X: pd.DataFrame):
        self.generator.eval()
        
        data = from_numpy(X.to_numpy()).to(self.device)
        with no_grad():
            present_mask = ~data.isnan()
            noise = rand_like(data) * 0.01
            with_noise = where(present_mask, data, noise)
                  
            pred = self.generator(with_noise, present_mask.to(dtype=int8))
            imputed = where(present_mask, data, pred)
        # print(imputed.isnan().any())
        
        return pd.DataFrame(imputed.cpu().numpy(), columns=X.columns, index=X.index)
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)
