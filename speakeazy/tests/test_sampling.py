import numpy as np 
import os

from .. import data,priors,sampling,fitting


class TestSampler():
    def setup(self):
         spec_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
         _spec_file = os.path.join(spec_dir, 
                                          'macs0417.1208_340.v0.spec.fits')
         spectrum = data.Data(spectrum_file=_spec_file,photometry_file=None,run_ID=1,phot_id=None)
         prs = priors.Priors(spectrum)
         
         prs.params['broadlines']=True
         
         fit_object = fitting.Fitter(spectrum,prs)
         fit_object.fit_redshift_chisq()
         #fit_object.plot_spectrum()
         self.sampler = sampling.Sampler(spectrum,prs,fit_object)
         


    def test_prior_sampling(self):
        """test prior sampling for setting up walkers
        
        - `speakeazy.sampling.Sampler.make_norm_prior`
        
        - `speakeazy.sampling.Sampler.sample_from_priors`
        """
        
        # test generation of gaussian priors 
        
        from scipy.stats import norm 
        
        N = 1000
        mu = 0.
        sigma = 1.
        norm_prior_dist = norm(loc=mu,scale=sigma)
        norm_prior_test = norm_prior_dist.rvs(size=N)
        
        norm_prior_code = self.sampler.make_norm_prior(sample=True)
        
        #test_mean = np.mean(norm_prior_test)
        #code_mean = np.mean(norm_prior_code)
        test_std = np.std(norm_prior_test)
        code_std = np.std(norm_prior_code)
        
        #test_mean_err = test_std/np.sqrt(N)
        #code_mean_err = code_std/np.sqrt(N)
        
        #print(test_mean,code_mean,test_std,code_std)
        
        #assert np.allclose(test_mean,code_mean,rtol=1e-1)
        assert np.allclose(test_std,code_std,rtol=1e-1)

        # test generation of walkers based on priors - FINISH THIS 
        #nwalkers = 100
        #nparam = 1
        #test_prior_matrix = np.zeros([nwalkers,nparam])
        
        #real_prior_matrix = self.sampler.sample_from_priors()
        
        #assert np.allclose(real_prior_matrix,test_prior_matrix,tol=1e-3)
        
        #return test_prior_matrix
        
        

        
    def test_walker_init(self):
         """test initialisation of parameter walkers for sampling with Emcee
        
        - `speakeazy.sampling.Sampler.init_walkers`
        """   
        
         walker_matrix = self.sampler.init_walkers()
         
         print(walker_matrix)
         
         self.sampler.run_emcee()
