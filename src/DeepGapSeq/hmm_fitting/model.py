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
            min_covar=100,
            tol = 0.0001,
            n_iter=1000,
            init_params="stmc",  # auto init all params
            algorithm="viterbi")

    def train_several_times(self,data,lengths,n_times):
        for i in range(n_times):
            remodel = self.model
            remodel.fit(data,lengths)
            if remodel.score(data,lengths) > self.model.score(data,lengths):
                self.model = remodel
        
    def fit(self,data,lengths,several_times=False):
        data = data - np.mean(data)
        data = data / np.std(data)
        self.model = self.model.fit(data,lengths)
        if several_times:
            self.train_several_times(data,lengths,10)

    def predict(self,data,lengths):
        data = data - np.mean(data)
        data = data / np.std(data)
        return self.model.predict(data,lengths)

    def get_performance(self,predicted_states, labels):
        score = 0
        for i in range(predicted_states.shape[0]):
            if predicted_states[i] == labels[i]:
                score += 1
        return score/len(predicted_states)    
        
class Model_selector():
    pass

