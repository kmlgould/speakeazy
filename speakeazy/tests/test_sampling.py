import numpy as np 

from .. import data,priors,sampling


class TestSampler():
    def setup(self):
         spectrum = data.Data(spectrum_file='../data/macs0417.1208_340.v0.spec.fits',photometry_file=None,run_ID=1,phot_id=None)
         prs = priors.Priors(spectrum)
         self.sampler = sampling.Sampler(spectrum,prs)


def test_prior_sampling(self):
    """test prior sampling for setting up walkers
    
    - `speakeazy.sampling.Sampler.make_norm_prior`
    
    - `speakeazy.sampling.Sampler.sample_from_priors`
    """
    
    # test generation of gaussian priors 
    
    from scipy.stats import norm 
    
    norm_prior_test = norm(loc=0.,scale=1.)
    norm_prior_sample = norm_prior_test.rvs(size=100)
    
    norm_prior = self.sampler.make_norm_prior(sampling=True)
    
    assert np.allclose(norm_prior_sample,norm_prior,tol=1e-3)

    # test generation of walkers based on priors - FINISH THIS 
    #nwalkers = 100
    #nparam = 1
    #test_prior_matrix = np.zeros([nwalkers,nparam])
    
    #real_prior_matrix = self.sampler.sample_from_priors()
    
    #assert np.allclose(real_prior_matrix,test_prior_matrix,tol=1e-3)
    
    #return test_prior_matrix
    
    

    
#def test_walker_init():
#     """test initialisation of parameter walkers for sampling with Emcee
#    
#    - `speakeazy.sampling.Sampler.init_walkers`
#    """   
