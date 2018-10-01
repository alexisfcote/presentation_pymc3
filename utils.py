from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import pymc3 as pm
from scipy import stats
import theano.tensor as t


def kde_scatter(x, y, colormap=None, size_factor=2, **kwargs):   
    if colormap is None:
        colormap = LinearSegmentedColormap.from_list(
        'sns_heat', [ sns.color_palette()[i] for i in [0, 3]], N=100)
    
    x = np.squeeze(x)
    y = np.squeeze(y)
    
    samples = np.vstack((x, y))
    
    samples = samples[:, np.all(~np.isnan(samples), axis=0)]
    
    densObj = kde( samples )

    def makeColours( vals ):
        colours = np.zeros( (len(vals),3) )
        norm = Normalize( vmin=vals.min(), vmax=vals.max() )

        #Can put any colormap you like here.
        colours = [cm.ScalarMappable( norm=norm, cmap=colormap).to_rgba( val ) for val in vals]

        return colours

    colours = makeColours( densObj.evaluate( samples ) )
    sizes   = densObj.evaluate( samples )
    sizes   = sizes / np.min(sizes) * size_factor

    return plt.scatter( samples[0], samples[1], color=colours, s=sizes, **kwargs)


def bayesian_correlation(vect1, vect2, robust=True):
    '''  Sample the posterior distribution of a multivariateStudent-t distribution observing vect1 and vect2.
         Compute the equivalent of pearsonr if robust=False, i.e. MvNormal.
    '''
    def covariance(sigma, rho):
        C = t.fill_diagonal(t.alloc(rho, 2, 2), 1.)
        S = t.diag(sigma)
        M = S.dot(C).dot(S)
        return M
    
    vect1 = np.reshape(vect1, (-1, 1))
    vect2 = np.reshape(vect2, (-1, 1))


    with pm.Model() as multivariate:
        # priors
        sigma1 = pm.HalfCauchy('sigma1', 2*np.std(vect1))
        sigma2 = pm.HalfCauchy('sigma2', 2*np.std(vect2))
        
        r = pm.Uniform('r', lower=-1, upper=1,
                         testval=stats.spearmanr(vect1, vect2)[0],  # init with Spearman's correlation
                      )

        cov   = pm.Deterministic('covl',   covariance(t.stack((sigma1, sigma2), axis=0), r))

        μ1 = pm.Normal('μ1', mu=np.mean(vect1), sd=2*np.std(vect1))
        μ2 = pm.Normal('μ2', mu=np.mean(vect2), sd=2*np.std(vect2))

        
        if not robust:
            mult_norm = pm.MvNormal('mult_norm', mu=[μ1, μ2],
                                    cov=cov, observed=np.hstack((vect1, vect2)))
        else: 
            num = pm.Exponential('nu_minus_one', lam=1. / 29., testval=1)
            ν = pm.Deterministic('ν', num + 1)
            mult_norm = pm.MvStudentT('mult_norm',nu=ν, mu=[μ1, μ2],
                                    cov=cov, observed=np.hstack((vect1, vect2)))
        
        trace = pm.sample()
    return trace

def summary(trace, varnames=None, plot_convergence_stats=False):
    pm.plot_posterior(trace, varnames=varnames)
    plt.show()
    pm.plots.traceplot(trace, varnames=varnames)
    plt.show()
    pm.plots.forestplot(trace, varnames=varnames)
    plt.show()
    if plot_convergence_stats:
        pm.plots.energyplot(trace)
        plt.show()
        print('Gelman-Rubin ', max(np.max(gr_values) for gr_values in pm.gelman_rubin(trace).values()))
    return pm.summary(trace, varnames=varnames)