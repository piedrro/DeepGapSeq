import abc
import pomegranate as pg
import numpy as np 
import hmmlearn.hmm as hmm


class Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def get_performance():
        pass

class HMM_pg(Model):
    def __init__(self, n_states=2,algorithem='baum-welch'):
        self.n_states = n_states
        self.algorithem = algorithem
        self.model = None

    def fit(self,data):
        return pg.HiddenMarkovModel.from_samples(
            pg.NormalDistribution,
            n_components=self.n_states,
            X=data,
            n_jobs=-1,
            algorithm=self.algorithem
            )
    def get_performance():
        
        
        pass

class HMM_learn():
    def __init__(self,n_states=2):
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            params="stmc",  # auto init all params
            algorithm="viterbi")

    def fit(self,data):
        data = data - np.mean(data)
        data = data / np.std(data)
        self.model = self.model.fit(data)
        

    def get_performance():
        pass

class Model_selector():
    pass