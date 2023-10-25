import os
import time
import warnings
from collections import OrderedDict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_MAIN_FREE"] = "1"


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits

import corner

import eazy
import emcee

import pathos.multiprocessing as mp

from grizli import utils


class Sampler(object):
    """Sampler 

    _extended_summary_

    Arguments:
        object -- _description_
    """
    def __init__(self,data,prior,fit_object):
        self.data = data
        self.prior = prior
        self.params = prior.params
        self.theta = fit_object.theta
        self.covar_i = fit_object.covar_i
    
    @staticmethod
    def make_norm_prior(mean=0.,sigma=1.,nwalkers=1000,sample=False):
        from scipy.stats import norm
        norm_prior = norm(loc=mean,scale=sigma)
        if sample:
            return norm_prior.rvs(size=nwalkers)
        else:
            return norm_prior
    
    def sample_from_priors(self,nwalkers,nparam,npa):
        """sample_from_priors _summary_

        given priors and parameters, sample from the prior distribution for each parameter
        """
        
        nparam_i = npa+self.params['epoly']+self.params['ppoly']
        
        prior_matrix = np.zeros([nwalkers,nparam_i])
        
        if self.data.grating != "prism":
            zscale = 0.0001

        else:
            zscale = 0.1
        
        prior_matrix[:,0] = self.make_norm_prior(self.params['zbest'],zscale,nwalkers,sample=True)
        prior_matrix[:,1] = self.prior.sc_rv.rvs(size=nwalkers)
        prior_matrix[:,2] = self.prior.vw_rv.rvs(size=nwalkers)
        
        if npa==4:
            prior_matrix[:,3]= self.prior.vw_b_rv.rvs(size=nwalkers) # but depends on if there are broadlines or not... 

        else:
            None 
        
        # something for epoly and ppoly here .... 
        
        # get coeffs for polyfit to straight line - which is average prior, then set coeffs on tight gaussians. 
        
        o = self.params['epoly'] - 1 # defined as ncoeffs instead of order, which is backwards and should be changed. 
        y = np.random.sample(size=len(self.data.spec_wobs))+2. # error scaling should be around 1-3. 
        c = np.polyfit(self.data.spec_wobs,y,o)
        
        for i in range(self.params['epoly']):
            prior_matrix[:,npa+i] = self.make_norm_prior(mean=c[i],sigma=0.1*abs(c[i]),nwalkers=nwalkers,sample=True)
            
        # doesn't include photometry scaling right now. 
        
        
        return prior_matrix
        
        
        
    
    def init_walkers(self,walkers_per_param=3,ball_size=1e-3):
        """init_walkers _summary_

        Takes parameter initial positions and priors and initialises the walker matrixes for the EMCEE run. 
        """
        
        # for scale disp, velocity widths, redshift, error scaling, photometry scaling, the walkers are random draws from the prior. 
        
        # for the line coefficients, the walkers are random draws from the covariance matrix from the least squares fit. 

        broadlines = self.params['broadlines']

        if broadlines:
        
            npa = 4
        else:
            npa = 3

            
        theta = self.theta.values() # parameter values 
        
        temps = [self.theta[t] for t in self.theta.keys() if (("line" in t) | ("bspl" in t))] # parameters which have covar. 

        nparam = len(theta) # Todo: this should be a property of the fitting class actually 
        nwalkers = nparam*walkers_per_param
    
        
        
        if self.covar_i is not None:
            print('initalising walkers using covar matrix')
            
            initial_walker_matrix = np.zeros([nwalkers,nparam])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # initialise line walker positions using covariance matrix from fit 
                initial_walker_matrix[:,npa+self.params['epoly']+self.params['ppoly']:] = np.random.multivariate_normal(temps, self.covar_i, size=nwalkers)
                # initalise all other parameters - all walkers drawn from prior distributions instead - todo
                
                # sample randomly from priors: 
                
                initial_walker_matrix[:,:npa+self.params['epoly']+self.params['ppoly']] = self.sample_from_priors(nwalkers,nparam,npa)
                
                #initial_walker_matrix[:,:npa+self.params['epoly']+self.params['ppoly']] = np.array(list(theta))[:npa+self.params['epoly']+self.params['ppoly']] + ptb_cv * np.random.randn(nwalkers, npa+self.params['epoly']+self.params['ppoly'])
                
                
                # make sure walkers are started with positive velocity width
                # Assign replacement values to the zero elements with some noise - draw again from the prior? 
                #initial_walker_matrix[:,1][initial_walker_matrix[:,1] < 0.] = np.random.choice(initial_walker_matrix[:,1][initial_walker_matrix[:,1] > 0.],size=len(initial_walker_matrix[:,1][initial_walker_matrix[:,1] < 0.]),replace=False) + 10.*np.random.randn(len(pos_cv[:,1][pos_cv[:,1] < 0.]))
                
        else:
            ptb = np.array(list(theta))*ball_size
            initial_walker_matrix = np.array(list(theta)) + ptb * np.random.randn(nwalkers, nparam)
            #fpos[:,1] = np.array(list(theta))[1] + ptb_cv[1]*np.random.gamma(nwalkers,1)
            
        return initial_walker_matrix
   
    # depracated
    def make_mcmc_draws(self,wp,init=[1e-3,0.2,1e-3,1e-2]):
        # wp = walkers per parameter
        # initalise walkers for emcee run 
        broadlines = self.params['broadlines']

        if broadlines:
        
            npa = 4
        else:
            npa = 3

            
        theta = self.theta.values()
        nparam = len(theta)
        temps = [self.theta[t] for t in self.theta.keys() if (("line" in t) | ("bspl" in t))]
        print(len(temps))
        print(len(self.covar_i))
        nwalkers = nparam*wp
    
        ptb = np.array(list(theta))*1e-3

        if len(self.prior_widths)>0:
            init_ball = self.prior_widths
        else:
            init_ball = init
        
        if self.covar_i is not None:
            print('initalising walkers using covar matrix')
            pos_cv = np.zeros([nwalkers,nparam])
            a = (np.array(list(theta))[:npa+self.params['epoly']+self.params['ppoly']])
            b = np.array(init_ball[:-1]+[init_ball[-1] for i in range(self.params['epoly']+self.params['ppoly'])])
            ptb_cv = np.array([i*j for (i,j) in zip(a,b)])
            ptb_cv[0] = init_ball[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mu = np.random.multivariate_normal(temps, self.covar_i, size=nwalkers)
        
                pos_cv[:,npa+self.params['epoly']+self.params['ppoly']:] = mu
                pos_cv[:,:npa+self.params['epoly']+self.params['ppoly']] = np.array(list(theta))[:npa+self.params['epoly']+self.params['ppoly']] + ptb_cv * np.random.randn(nwalkers, npa+self.params['epoly']+self.params['ppoly'])
                # make sure walkers are started with positive velocity width
                vw_mask = pos_cv[:,1]<0.
                #pos_cv[:,1][vw_mask] = np.array(list(theta))[1] 
                m = np.nanmedian(pos_cv[:,1][pos_cv[:,1] > 0.])
                # Assign replacement values to the zero elements with some noise
                pos_cv[:,1][pos_cv[:,1] < 0.] = np.random.choice(pos_cv[:,1][pos_cv[:,1] > 0.],size=len(pos_cv[:,1][pos_cv[:,1] < 0.]),replace=False) + 10.*np.random.randn(len(pos_cv[:,1][pos_cv[:,1] < 0.]))
                fpos = pos_cv
        else:
            fpos = np.array(list(theta)) + ptb * np.random.randn(nwalkers, nparam)
            #fpos[:,1] = np.array(list(theta))[1] + ptb_cv[1]*np.random.gamma(nwalkers,1)
            
        return fpos


    def log_prob_data_global(self,theta):

        # initialise parameters

        broadlines = self.params['broadlines']

        if broadlines:
        
            npa = 4
            z, vw, vw_b, sc = theta[:npa]
        else:
            npa = 3
            z, vw, sc = theta[:npa]
            vw_b = 0.


        
        escale_coeffs = theta[npa:npa+self.params['epoly']]
        pscale_coeffs = theta[npa+self.params['epoly']:npa+self.params['epoly']+self.params['ppoly']]
        line_coeffs = theta[npa+self.params['epoly']+self.params['ppoly']:npa+self.params['epoly']+self.params['ppoly']+self.params['nlines']] # linecoeffs
        bline_coeffs = theta[npa+self.params['epoly']+self.params['ppoly']+self.params['nlines']:npa+self.params['epoly']+self.params['ppoly']+self.params['nlines']+self.params['nblines']] # linecoeffs
        cont_coeffs = theta[npa+self.params['epoly']+self.params['ppoly']+self.params['nlines']+self.params['nblines']:] # the rest are nspline coeffs 
        
        coeffs = theta[npa+self.params['epoly']+self.params['ppoly']:] #line and spline coeffs. 

        #print(len(coeffs))
      
        #print(pscale_coeffs)
        # make model for continuum and lines

        templ_arr,tline = self.generate_templates(z,sc,vw,vw_b,init=False)

        

        nline_mask = tline==0

        if broadlines:
            nbline_mask = tline==1
        

        mspec = templ_arr.T.dot(coeffs)
        #_mline = _A.T.dot(coeffs*tline) #+ _A.T.dot(coeffs*tbline) #broad
        _mline = templ_arr.T.dot(coeffs*nline_mask)

        if broadlines:
            _mbline = templ_arr.T.dot(coeffs*nbline_mask)
                          
            _mcont = mspec - _mline - _mbline

        else:
            _mcont = mspec - _mline
            _mbline = 0.
            
        xr = spec_wobs
        xr_ang = xr*1e4

        
        mask = valid
        flam = spec_fnu*to_flam
        eflam = spec_efnu*to_flam

        
        
        escale = (np.polyval(escale_coeffs,xr))
        #print(np.nanmedian(escale))
        e2 = (eflam*escale)**2. # scale errors in fnu, THEN convert to flam??



        if self.params['scale_p']==False:
            pscale = 1.
            lnphot = 0.
        else:
            # scale to photometry
            pscale = (np.polyval(pscale_coeffs,xr))
            p2 = (self.phote_flam)**2.
            
            mspec_c = (1e-19*mspec*pscale)
            temp_spec = eazy.templates.Template(arrays=(xr_ang[mask],mspec_c[mask]),name='temp2')
            mphots = []
            for fil in self.filters:
                mphot = (temp_spec.integrate_filter(filt=fil,flam=True))
                mphots.append(mphot)
            xr_phot = np.array([9022.922,11543.009,19886.479,27577.959,35682.277,40820.574,44036.71])/1e4
            lnphot = -0.5 * (((self.phot_flam - mphots) ** 2. / p2) + (np.log(2.*np.pi*p2))).sum()

        lnp =  -0.5 * (((flam - mspec) ** 2. / e2) + (np.log(2.*np.pi*e2)))[mask].sum() #removed neg, makes it -ln(P), minimize
        if broadlines:
            logprior = self.z_rv.logpdf(z) + self.escale_prior(escale) + self.sc_prior(sc) + self.vw_prior(vw) + self.vwb_prior(vw_b) + self.coeffs_prior(coeffs) #+self.balmer_ratios_prior(line_fluxes)
        else:
            logprior = self.z_rv.logpdf(z) + self.escale_prior(escale) + self.sc_prior(sc) + self.vw_prior(vw)
        lprob = lnp + logprior + lnphot

        if np.isnan(lprob):
            lprob = -np.inf
        else:
            lprob = lprob

        return lprob
    
    def run_emcee(self,n_it=100,wp=3,nds=8,zin=1e-3,mp=False):
        
#         snx = self.theta.values()
        
#         # initialise walkers 
        
#         lj = len(snx)
#         pos = np.array(list(snx)) + 1e-8 * np.random.randn(lj*3, lj)
#         # instead sample from covariance draws 
#         pos = np.array(list(snx))

        pos = self.make_mcmc_draws(wp=wp,init=[zin,0.2,1e-3,1e-2])
        print(pos)
        nwalkers, ndim = pos.shape
        print(nwalkers,ndim)
        
        # Set up the backend
        # Don't forget to clear it in case the file already exists
        filename = self.ID+"_emcee_run.h5"
        print(filename)
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)

        
        if mp:
            
             # Pool gets stuck with class methods because it uses pickle, which can't pickle instances. 
             # this is an attempted work around, using dill, which can serialize basically anything. 
            #import multiprocessing_on_dill as multiprocessing
            #from multiprocessing_on_dill import Pool
            #multiprocessing.set_start_method("fork")
            #os.system("taskset -p 0xff %d" % os.getpid())
            import pathos.multiprocessing as multiproc
            mp_pool = multiproc.ProcessPool(nodes=nds)
            #sampler = emcee.EnsembleSampler(nwalkers, ndim, sp.log_likelihood, args = (sp,0), pool=pool)
            #start = time.time()
            #sampler.run_mcmc(pos, n_it, progress=True)
            #end = time.time()
            #multi_time = end - start
            #print("Multiprocessing took {0:.1f} seconds".format(multi_time))
           
            with mp_pool as pool:
                
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob_data_global, pool=pool,backend=backend)
                #start = time.time()
                #sampler.run_mcmc(pos, n_it, progress=True)
                #end = time.time()
                #multi_time = end - start
                #print("Multiprocessing took {0:.1f} seconds".format(multi_time))
                
                max_n = n_it

                # We'll track how the average autocorrelation time estimate changes
                #index = 0
                #autocorr = np.empty(max_n)
                #std_params

                # This will be useful to testing convergence
                old_stdp = np.inf
                old_accept_frac = 1.0

                # Now we'll sample for up to max_n steps
                start = time.time()
                for sample in sampler.sample(pos, iterations=max_n, progress=True):
                    # Only check convergence every 100 steps, advance the chain 
                    if sampler.iteration % 100:
                        continue

                    chain = sampler.get_chain()
                    stdp = np.nanstd(chain[-1, :, :], axis=0) #calc stdf for all params 
                    accept_frac = np.mean(sampler.acceptance_fraction)
                    #autocorr[index] = np.mean(tau)
                    #index += 1

                    # Check convergence
                    #converged = np.all(tau * 100 < sampler.iteration)
                    converged = np.all(np.abs(old_stdp - stdp) / stdp < 0.1)
                    #converged &= 0.2<accept_frac<0.25
                    #print("tau diff", np.abs(old_tau - tau) / tau)
                    if converged:
                        print("Converged, stopping chain")
                        break
                    old_stdp = stdp
                    old_accept_frac = accept_frac
                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
                #print("{0:.1f} times faster than serial".format(serial_time / multi_time))

        else: 

            sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, sp.log_likelihood, args = (sp,0))
            sampler.run_mcmc(pos, n_it, progress=True);
        
        #flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
        #products = OrderedDict()
        #labels = list(self.theta.keys())

        #for i,l in zip(range(ndim),labels):
        #    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        #    q = np.diff(mcmc)
            
        #    products[l] = [mcmc[1], q[0], q[1]] # write results 
       #hickle.dump(products,self.ID+'_emcee_products.dict') 

        
        return sampler
    
    def read_samples(self,burnin,thin,flat=True,plot_walkers=True):
        filename = self.ID+"_emcee_run.h5"
        reader = emcee.backends.HDFBackend(filename)
        if plot_walkers:
            self.plot_walkers(reader)
        if flat:
            samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
        else:
            samples = reader.get_chain()
        return samples
    
    def plot_models(self,flat_samples,nmodels=100):
            params = np.nanmean(flat_samples, axis=0)
            mspec = sp.log_likelihood(params,sp,2)
            self.model_spec = mspec
            inds = np.random.randint(len(flat_samples), size=nmodels)
            self.plot_spectrum(save=True,fname=self.ID+"_emcee_fullfit.png",flat_samples=flat_samples[inds])

    def plot_walkers(self,sampler):
        samples = sampler.get_chain()
        labels = list(sp.theta.keys())
        for i in range(len(labels)):
            plt.figure(figsize=(10,3))
            plt.plot(samples[:, :, i], "cornflowerblue", alpha=0.1)
            medf = np.nanmedian(samples[:, :, i], axis=1)
            stdf = np.nanstd(samples[:, :, i], axis=1)
            print(samples[:, :, i].shape)
            plt.plot(range(len(medf)),medf, "midnightblue", alpha=0.9)
            plt.xlim(0, len(samples))
            plt.ylabel(labels[i])
            plt.xlabel("step number");
            plt.show()


    def plot_corner(self,flat_samples,labs=['z','vw','vw_b','sc', 'line Ha','line NII'],save=False):
        #labs=['z','vw','sc', 'line Hb','line OIII-4959','line OIII-5007']
        labels = list(self.theta.keys())
        indexes = [index for index in range(0,len(labels)) if labels[index] in labs]
           
        cfig = corner.corner(
            flat_samples[:, indexes], labels=labs)

        if save:
            cfig.savefig(f'{self.ID}_corner.png')

    def plot_err_scale(self,flat_samples,nmodels):
            
        escale_coeffs = np.array([self.theta[f'escale_{i}'] for i in range(self.params['epoly'])])
        escale = (np.polyval(escale_coeffs,spec_wobs))
        labels = list(self.theta.keys())

    
        labs = np.array([f'escale_{i}' for i in range(self.params['epoly'])])
        indexes = [index for index in range(len(labels)) if labels[index] in labs]
        inds = np.random.randint(len(flat_samples), size=nmodels)
        plt.figure(figsize=(12,4))
        plt.plot(spec_wobs,escale,lw=2,color='red')
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel('Error scaling')
        for sample in flat_samples[inds]:
    
            e_coeff = sample[indexes]
            escale_i = (np.polyval(e_coeff,spec_wobs))
            plt.plot(spec_wobs,escale_i,lw=1,alpha=0.1,color='red')
    
            plt.xlabel(r'Wavelength [$\mu$m]')
            plt.ylabel('Error scaling')
        plt.show()

    def make_results(self,sampler,burnin,thin):
        flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
        products = OrderedDict()
        labels = list(self.theta.keys())

        for i,l in zip(range(ndim),labels):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            
            products[l] = [mcmc[1], q[0], q[1]] # write results 
        hickle.dump(products,self.ID+'_emcee_products.dict') 

        return products

