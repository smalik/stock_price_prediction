from pymssa import MSSA
from prince import PCA
import pandas as pd
import numpy as np
#import multipledispatch as mdisp

class TickerTransform(object):

    def __init__(self, data:pd.Series):
        self.data = data
        self.model_ssa = None
        self.model_pca = None

    def get_ssa(self, ncomp:int= None, wsize= 60):
        model = MSSA(n_components= ncomp,
                     window_size= wsize,
                     verbose= True)
        model.fit(self.data)

        self.model_ssa = model

    def retrieve_signal(self, explained_variance:float = 0.9) -> pd.DataFrame:
        ind = np.sum(np.cumsum(self.model_ssa.explained_variance_ratio_) < explained_variance)
        print(f'ind: {ind}')
        if ind == 0:
            ind = 1
        print(f'ind: {ind}')
        components = self.model_ssa.components_[0, :, ind]
        print(f'Components shape is: {components.shape}')
        signal = pd.DataFrame(components, index= self.data.index, columns= ['price_signal']).sum(axis=1)
        return signal

    def get_pca(self, components:int = 3):

        data = self.df
        results = dict()
        pca = PCA(n_components = components,
                  n_iter=100,
                  rescale_with_mean = True,
                  rescale_with_std = True,
                  copy = True,
                  check_input = True
                  )
        results['fit'] = pca.fit(data)
        results['rotated'] = pca.fit_transform(data)
        results['feature_correlations'] = fit.column_correlations(data)

        return results
