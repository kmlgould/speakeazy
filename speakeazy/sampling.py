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
import hickle
import re

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
        self.model = fit_object.model
        self.model.spl_mask = fit_object.spl_mask
    
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

        if self.params['z_in'] is not None:
            prior_matrix[:,0] = self.prior.z_rv.rvs(size=nwalkers)
        else:
            if self.data.grating != "prism":
                zscale = 0.0001
    
            else:
                zscale = 0.01
                
            self.prior.z_rv = self.make_norm_prior(self.params['zbest'],zscale,sample=False)
            prior_matrix[:,0] = self.make_norm_prior(self.params['zbest'],zscale,nwalkers,sample=True)

        prior_matrix[:,1] = self.prior.vw_rv.rvs(size=nwalkers)
        #prior_matrix[:,2] = self.prior.sc_rv.rvs(size=nwalkers)
        
        if npa==4:
            prior_matrix[:,2]= self.prior.vwb_rv.rvs(size=nwalkers) # but depends on if there are broadlines or not... 
            prior_matrix[:,3] = self.prior.sc_rv.rvs(size=nwalkers)
        else:
            prior_matrix[:,2] = self.prior.sc_rv.rvs(size=nwalkers)
        
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
                print(initial_walker_matrix)
                
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



    def generate_model(self,theta):
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
        line_coeffs = theta[npa+self.params['epoly']+self.params['ppoly']:npa+self.params['epoly']+self.params['ppoly']+self.model.nlines] # linecoeffs
        bline_coeffs = theta[npa+self.params['epoly']+self.params['ppoly']+self.model.nlines:npa+self.params['epoly']+self.params['ppoly']+self.model.nlines+self.model.nblines] # linecoeffs
        cont_coeffs = theta[npa+self.params['epoly']+self.params['ppoly']+self.model.nlines+self.model.nblines:] # the rest are nspline coeffs 
        
        coeffs = theta[npa+self.params['epoly']+self.params['ppoly']:] #line and spline coeffs. 

        # make model for continuum and lines

        templ_arr,tline = self.model.generate_templates(self.data,z,sc,vw,vw_b,self.params['blue_in'],self.theta,chisq=False,broadlines=self.params['broadlines'])

        

        nline_mask = tline==0

        if broadlines:
            nbline_mask = tline==1
        

        mspec = templ_arr.T.dot(coeffs)
        #_mline = _A.T.dot(coeffs*tline) #+ _A.T.dot(coeffs*tbline) #broad
        _mline = templ_arr.T.dot(coeffs*nline_mask)
        #_Acont = (templ_arr.T*coeffs)[mask,:][:,self.model.nlines+self.model.nblines:]
        #_Acont[_Acont < 0.001*_Acont.max()] = np.nan

        if broadlines:
            _mbline = templ_arr.T.dot(coeffs*nbline_mask)
                          
            _mcont = mspec - _mline - _mbline

        else:
            _mcont = mspec - _mline
            _mbline = 0.
            
        return mspec,_mline
        
        
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
        line_coeffs = theta[npa+self.params['epoly']+self.params['ppoly']:npa+self.params['epoly']+self.params['ppoly']+self.model.nlines] # linecoeffs
        bline_coeffs = theta[npa+self.params['epoly']+self.params['ppoly']+self.model.nlines:npa+self.params['epoly']+self.params['ppoly']+self.model.nlines+self.model.nblines] # linecoeffs
        cont_coeffs = theta[npa+self.params['epoly']+self.params['ppoly']+self.model.nlines+self.model.nblines:] # the rest are nspline coeffs 
        
        coeffs = theta[npa+self.params['epoly']+self.params['ppoly']:] #line and spline coeffs. 

        # make model for continuum and lines

        templ_arr,tline = self.model.generate_templates(self.data,z,sc,vw,vw_b,self.params['blue_in'],self.theta,chisq=False,broadlines=self.params['broadlines'])

        

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
            
        xr = self.data.spec_wobs
        xr_ang = xr*1e4

        
        mask = self.data.valid
        flam = self.data.spec_fnu*self.data.to_flam
        eflam = self.data.spec_efnu*self.data.to_flam

        
        
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
        
         # h line prior
        hlr = ['line Hb','line Hg', 'line Hd', 'line Ha', 'line NII']
        labels = list(self.theta.keys())
        indexes = [index for index in range(len(labels)) if labels[index] in hlr]
        line_fluxes = theta[indexes]
        if broadlines:
            logprior = self.prior.z_rv.logpdf(z) + self.prior.escale_prior(escale) + self.prior.sc_prior(sc) + self.prior.vw_prior(vw) + self.prior.vwb_prior(vw_b) +self.prior,balmer_ratios_prior(line_fluxes)
        else:
            logprior = self.prior.z_rv.logpdf(z) + self.prior.escale_prior(escale) + self.prior.sc_prior(sc) + self.prior.vw_prior(vw) + self.prior.coeffs_prior(coeffs) +self.prior.balmer_ratios_prior(line_fluxes)
        lprob = lnp + logprior + lnphot

        if np.isnan(lprob):
            lprob = -np.inf
        else:
            lprob = lprob

        return lprob
    
    def run_emcee(self,n_it=10,wp=3,nds=4,zin=1e-3,conv_frac=0.05,mp=True):
        
#         snx = self.theta.values()
        
#         # initialise walkers 
        
#         lj = len(snx)
#         pos = np.array(list(snx)) + 1e-8 * np.random.randn(lj*3, lj)
#         # instead sample from covariance draws 
#         pos = np.array(list(snx))

        pos = self.init_walkers(walkers_per_param=wp)
        print(pos)
        nwalkers, ndim = pos.shape
        print(nwalkers,ndim)
        
        # Set up the backend
        # Don't forget to clear it in case the file already exists
        backend = emcee.backends.HDFBackend(f'{self.data.run_ID[1:]}/emcee_run.h5')
        backend.reset(nwalkers, ndim)

        
        if mp:
            import pathos.multiprocessing as multiproc
            mp_pool = multiproc.ProcessPool(nodes=nds)
         
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
                    if sampler.iteration % 500:
                        continue

                    chain = sampler.get_chain()
                    stdp = np.nanstd(chain[-1, :, :], axis=0) #calc stdf for all params 
                    accept_frac = np.mean(sampler.acceptance_fraction)
                    #autocorr[index] = np.mean(tau)
                    #index += 1

                    # Check convergence
                    #converged = np.all(tau * 100 < sampler.iteration)
                    converged = np.all(np.abs(old_stdp - stdp) / stdp < conv_frac)
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
                    nwalkers, ndim, self.log_prob_data_global)
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
        filename = str(self.data.run_ID)+"_emcee_run.h5"
        reader = emcee.backends.HDFBackend(filename)
        if plot_walkers:
            self.plot_walkers(reader)
        if flat:
            samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
        else:
            samples = reader.get_chain()
        return samples
    
    # fix this 
    def plot_models(self,flat_samples,nmodels=10):
            params = np.nanmean(flat_samples, axis=0)
            mspec,mline = self.generate_model(params)
            self.model_spec = mspec
            self.model_line = mline
            inds = np.random.randint(len(flat_samples), size=nmodels)
            self.simple_plot_spectrum(save=True,flat_samples=flat_samples[inds])

    def plot_walkers(self,sampler):
        samples = sampler.get_chain()
        labels = list(self.theta.keys())
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
            cfig.savefig(f'{self.data.ID}_corner.png')

    # fix this 
    def plot_err_scale(self,flat_samples,nmodels):
            
        escale_coeffs = np.array([self.theta[f'escale_{i}'] for i in range(self.params['epoly'])])
        escale = (np.polyval(escale_coeffs,self.data.spec_wobs))
        labels = list(self.theta.keys())

    
        labs = np.array([f'escale_{i}' for i in range(self.params['epoly'])])
        indexes = [index for index in range(len(labels)) if labels[index] in labs]
        inds = np.random.randint(len(flat_samples), size=nmodels)
        plt.figure(figsize=(12,4))
        plt.plot(self.data.spec_wobs,escale,lw=2,color='red')
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel('Error scaling')
        for sample in flat_samples[inds]:
    
            e_coeff = sample[indexes]
            escale_i = (np.polyval(e_coeff,self.data.spec_wobs))
            plt.plot(self.data.spec_wobs,escale_i,lw=1,alpha=0.1,color='red')
    
            plt.xlabel(r'Wavelength [$\mu$m]')
            plt.ylabel('Error scaling')
        plt.show()

    def make_results(self,sampler,burnin,thin):
        flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
        products = OrderedDict()
        labels = list(self.theta.keys())
        ndim = len(labels)

        for i,l in zip(range(ndim),labels):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            
            products[l] = [mcmc[1], q[0], q[1]] # write results 
        hickle.dump(products,str(self.data.run_ID)+'_emcee_products.dict') 

        return products

    def simple_plot_spectrum(self,save=True,fname=None,flat_samples=None,lines=False,ylims=None,xlims=None):
        
        mask = self.data.valid
        flam = self.data.spec_fnu*self.data.to_flam
        eflam = self.data.spec_efnu*self.data.to_flam
        
        flam[~mask] = np.nan
        eflam[~mask] = np.nan
        
        wav = self.data.spec_wobs
        
        xmin = np.nanmin(wav[mask])
        xmax = np.nanmax(wav[mask])

        fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,4))

        scale=1.
        
        if (self.prior.params['scale_p']==True):
            if (self.theta['pscale_0']!=1.) : 
                pscale_coeffs = np.array([self.theta[f'pscale_{i}'] for i in range(self.prior.params['ppoly'])])
                scale = (np.polyval(pscale_coeffs,wav))
            #else:
            #    scale = 1.

        #if scale==None:
        #    scale=1.

        scale = 1.     
        # plot data     
        plt.step(wav[mask],(flam*scale)[mask],color='grey',alpha=0.5,label='1D spectrum')
        ymax = 1.1*np.nanmax((flam*scale)[mask])
        ymin = np.nanmin((flam*scale)[mask])
        #print(ymin,ymax)
          
        if flat_samples is not None:
            for sample in flat_samples:
                mspec,mline = self.generate_model(sample)
                if lines:
                    plt.plot(wav,mline,color='cornflowerblue',lw=0.5,alpha=0.3)
                else:
                     plt.plot(wav,mspec,color='cornflowerblue',lw=0.5,alpha=0.3)                   
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel(r'F$_{\lambda}$ [10$^{-19}$ erg/s/cm$^{2}/\AA$]')
        zb = self.theta['z']
        #plt.title(f'{self.data.fname}, zspec={zb:.3f}',fontsize=10)
        plt.text(x=0.6,y=0.8,s=f'{self.data.run_ID}\n z={zb:.3f}',
                                 bbox = dict(facecolor = 'white', alpha = 0.5),fontsize=10,
                 transform=ax.transAxes)
        #plt.grid(alpha=0.6)
        if ylims is not None:
            plt.ylim(ylims[0],ylims[1])
        else:
            plt.ylim(-0.05,ymax)
            
        if xlims is not None:
            #print(xlims[0],xlims[1])
            plt.xlim(xlims[0],xlims[1])
        else:
            plt.xlim(xmin+0.05,xmax-0.03)
        if save:
            plt.savefig(f'{self.data.ID}/_fit.png')
        return #fig