import os

import re
from importlib import reload

import hickle as hkl
import msaexp
import numpy as np
from astropy.table import Table
from msaexp import pipeline, spectrum

print(f'msaexp version = {msaexp.__version__}')
print(f'numpy version = {np.__version__}')
import sys
import time
import warnings
from collections import OrderedDict
from functools import wraps

import astropy.units as u
import corner
import dill
import eazy
import emcee
import matplotlib.pyplot as plt
import msaexp.resample_numba
import msaexp.spectrum
import numba
import pathos.multiprocessing as mp
from astropy.io import fits
from grizli import utils
from grizli import utils as utils
from scipy import stats
from scipy.optimize import nnls
from tqdm import tqdm

utils.set_warnings()

from pylab import *

from .models import *
from .priors import * 
from .data import *
from .plotting import *

rc('axes', linewidth=1.5)
rc('xtick',direction='in')#, minor_visible=True, major_width=1.2, minor_width=1.0)
rc('ytick',direction='in')#, minor_visible=True, major_width=1.2, minor_width=1.0)
rc('font',size=14)
plt.rcParams["font.family"] = "serif"

class Fitting(object):
    """Fitting 

    _extended_summary_

    Arguments:
        object -- _description_
    """
    
    def __init__(self) -> None:
        pass
    
       
    def __fit_redshift(self,zgrid,zstep,sc,vw,vw_b,zfix=None): 
        
        mask = Data.valid
        flam = Data.spec_fnu*Data.to_flam
        eflam = Data.spec_efnu*Data.to_flam


        flam[~mask] = np.nan
        eflam[~mask] = np.nan
        
        
        if isinstance(zfix, float): 
            
            _A,line_names_,broad_line_names_,tline = Models.generate_templates(zfix,sc,vw,vw_b,init=True)
        
            okt = _A[:,mask].sum(axis=1) > 0
            _Ax = _A[okt,:]/eflam
            _yx = flam/eflam
            #_x = np.linalg.lstsq(_Ax[:,mask].T, 
            #                         _yx[mask], rcond=None)
            _x = nnls(_Ax[:,mask].T, 
                                 _yx[mask])

            coeffs = np.zeros(_A.shape[0])
            coeffs[okt] = _x[0]
            
           # try:
            oktemp = okt & (coeffs != 0)

            print(np.sum(oktemp),'oktemp')

            AxT = (_A[oktemp,:]/eflam)[:,mask].T

            covar_i = utils.safe_invert(np.dot(AxT.T, AxT))
            covar = utils.fill_masked_covar(covar_i, oktemp)
            covard = np.sqrt(covar.diagonal())
                
            # save covariance matrix for emcee run 
            # only want to keep the lines that have non zero coeffs. 
            #cv_mask = list(np.ones(Priors.params['nspline'],dtype='bool')).append(coeffs[:Priors.params['nlines']]!=0)
            # spl_list = list(np.ones(Priors.params['nspline'],dtype='bool'))
            # cv_mask0 = list(coeffs[:Priors.params['nlines']+Priors.params['nblines']]!=0)
            # print(len(cv_mask0))
            # cv_mask = [*cv_mask0,*spl_list]
            # print(cv_mask)
            # print(covar_i.shape)
            #cv_mask = cv_mask.append(#bug??? wron way round???
            #covar_i_masked = covar_i[cv_mask,cv_mask]
            self.covar_i =  np.squeeze(covar_i)
                

           # except:
           #     covard = coeffs*0.
            #    N = len(line_names_)
           #     covar = np.eye(N, N)
                

            
            return _A, coeffs, covard, line_names_, broad_line_names_, tline
        
        else:
            
        
            chi2 = zgrid*0.


            for iz, z in tqdm(enumerate(zgrid)):

                _A,line_names_,broad_line_names_,tline = Models.generate_templates(z,sc,vw,vw_b,init=True)

                okt = _A[:,mask].sum(axis=1) > 0
                _Ax = _A[okt,:]/eflam
                _yx = flam/eflam
                #_x = np.linalg.lstsq(_Ax[:,mask].T, 
                #                         _yx[mask], rcond=None)
                _x = nnls(_Ax[:,mask].T, 
                                 _yx[mask])


                coeffs = np.zeros(_A.shape[0])
                coeffs[okt] = _x[0]
                _model = _A.T.dot(coeffs)

                chi = (flam - _model) / eflam

                chi2_i = (chi[mask]**2).sum()
                chi2[iz] = chi2_i
        
            return zgrid, chi2
        
    def init_params(self):
        # initialise input params for log-likelihood fitting
           # could also include initial redshift fit? or we put that somewhere else?"""
        # return Priors.params which has params either fixed or variable 
        # if fixed, it has the option of which ones to fix and at which values 

        Priors.set_params()

        escale_coeffs = np.ones(Priors.params['epoly']) #npoly 
        
        if Priors.params['scale_p']==True:
            pscale_coeffs = np.ones(Priors.params['ppoly']) # np.ones(ppoly)
        else:
            pscale_coeffs = None
            
        Models.initialize_bsplines() #makes continuum splines
        
        
        is_prism = Data.grating in ['prism']

        # FIT REDSHIFT 

            # if input z is given, use that, otherwise use redshift grid 
        if Priors.params['z_in'] is not None:
            print("fix redshift", Priors.params['z_in'])
            Priors.params['zbest']=Priors.params['z_in']
            zbest = Priors.params['z_in']
            zgrid = 1
            zstep = 1
            _A, coeffs, covard, line_names_,broad_line_names_,tline = self.__fit_redshift(zgrid,zstep,Priors.params['sc'],Priors.params['vw'],Priors.params['vw_b'],zfix=Priors.params['z_in'])

        else: 
        
            if zstep is None:
                if (is_prism):
                    step0 = 0.002
                    step1 = 0.0001
                else:
                    step0 = 0.001
                    step1 = 0.00002
            else:
                step0, step1 = zstep
    
            zgrid = utils.log_zgrid(Priors.params['z_range'], step0)
            
            zg0, chi0 = self.__fit_redshift(zgrid,zstep,Priors.params['sc'],Priors.params['vw'],Priors.params['vw_b'],zfix=None)
            
            zbest0 = zg0[np.argmin(chi0)]
            
            # Second pass
            zgrid1 = utils.log_zgrid(zbest0 + np.array([-0.005, 0.005])*(1+zbest0), 
                                step1)
            
            zg1, chi1 = self.__fit_redshift(zgrid1,zstep,Priors.params['sc'],Priors.params['vw'],Priors.params['vw_b'],zfix=None)
            
            zbest = zg1[np.argmin(chi1)]
            
            Priors.params['zbest']=zbest
        
            _A, coeffs, covard, line_names_, broad_line_names_,tline = self.__fit_redshift(zgrid1,zstep,Priors.params['sc'],Priors.params['vw'],Priors.params['vw_b'],zfix=zbest)
        
        self.templates = _A
        #self.tline = tline

        nline_mask = tline==0
        nbline_mask = tline==1
        

        _model = _A.T.dot(coeffs)
        _mline = _A.T.dot(coeffs*nline_mask)
        if Priors.params['broadlines']:
            _mbline = _A.T.dot(coeffs*nbline_mask)        
            _mcont = _model - _mline - _mbline
            self.model_bline = _mbline
        else:
            _mcont = _model - _mline
        
        self.model_spec = _model
        self.model_line = _mline
        self.model_cont = _mcont

        mask = Data.valid
        flam = Data.spec_fnu*Data.to_flam
        eflam = Data.spec_efnu*Data.to_flam


        flam[~mask] = np.nan
        eflam[~mask] = np.nan

        full_chi2 = ((flam - _model)**2./eflam**2.)[mask].sum()
        cont_chi2 = ((flam - _mcont)**2./eflam**2.)[mask].sum()

        print(f'full chi2 = {full_chi2}, cont chi2 = {cont_chi2}')

        _Acont = (_A.T*coeffs)[mask,:][:,Priors.params['nlines']+Priors.params['nblines']:]
        _Acont[_Acont < 0.001*_Acont.max()] = np.nan

        self.Acont = _Acont

        
        line_coeffs = coeffs[:Priors.params['nlines']]
        cont_coeffs = coeffs[Priors.params['nlines']+Priors.params['nblines']:]
        line_covard = covard[:Priors.params['nlines']]
        cont_covard = covard[Priors.params['nlines']+Priors.params['nblines']:]   

        if Priors.params['broadlines']:

            bline_coeffs = coeffs[Priors.params['nlines']:Priors.params['nlines']+Priors.params['nblines']]
            bline_covard = covard[Priors.params['nlines']:Priors.params['nlines']+Priors.params['nblines']]
        
        theta_dict = OrderedDict()
        
        theta_dict['z'] = zbest
        theta_dict['vw'] = Priors.params['vw']
        if Priors.params['broadlines']:
            theta_dict['vw_b'] = Priors.params['vw_b']
        theta_dict['sc']=Priors.params['sc']
        
        escale_names = [f'escale_{x}' for x in range(Priors.params['epoly'])]
        for name,coeff in zip(escale_names,escale_coeffs):
            theta_dict[name]=coeff

            
        if Priors.params['scale_p']==True:
            pscale_names = [f'pscale_{x}' for x in range(Priors.params['ppoly'])]
            for name,coeff in zip(pscale_names,pscale_coeffs):
                theta_dict[name]=coeff

        
        bspl_names = [f'bspl_{x}' for x in range(len(cont_coeffs))]
        
         # remove line coeffs that are zero 
        
        line_mask = line_coeffs!=0. 
        Priors.params['nlines']=np.sum(line_mask)


        if Priors.params['broadlines']:
            bline_mask = bline_coeffs!=0. 
            Priors.params['nblines']=np.sum(line_mask)


        spl_mask = cont_coeffs!=0. 
        Priors.params['nspline']=np.sum(spl_mask)
        self.spl_mask = spl_mask

        for name,coeff in zip(np.array(line_names_)[line_mask].tolist(),line_coeffs[line_mask]):
            theta_dict['line '+name]=coeff

        if Priors.params['broadlines']:

            for name,coeff in zip(np.array(broad_line_names_)[bline_mask].tolist(),bline_coeffs[bline_mask]):
                theta_dict['broad line '+name]=coeff
            
        for name,coeff in zip(np.array(bspl_names)[spl_mask].tolist(),cont_coeffs[spl_mask]):
            theta_dict[name]=coeff
                  
        self.theta = theta_dict

        line_tab = OrderedDict()
        

        for name,coeff,covard in zip(np.array(line_names_)[line_mask].tolist(),line_coeffs[line_mask],line_covard[line_mask]):
            line_tab['line '+name]=[coeff,covard]

        self.line_table = line_tab

        if Priors.params['broadlines']:

            bline_tab = OrderedDict()
            
            for name,coeff,covard in zip(np.array(broad_line_names_)[bline_mask].tolist(),bline_coeffs[bline_mask],bline_covard[bline_mask]):
                bline_tab['broad line '+name]=[coeff,covard]
    
            self.bline_table = bline_tab
            
    def fit_spectrum(self,ll=False,**kwargs):
        """ initialises parameters and minimises -log likelihood """
            
        from scipy.optimize import minimize
        np.random.seed(42)

        self.init_params()
        for key in self.theta:
            print(key,":", self.theta[key])
        initial_params = np.array(list(self.theta.values()))
        
        Plotting.plot_spectrum(fname=self.ID+"initial_fit.png",**kwargs)
        
        
        
        if ll:

            Priors.init_logprior()
            llf = self.ID+'_llh_theta.dict'
            print(llf)
            
            # check likelihood fitting hasn't already been run, saves a bit of time 
            if os.path.isfile(llf): 
                saved_dict = hkl.load(llf)
                # check they keys are the same, in case you change the params e.g. epoly for fit_spectrum but forget to update the run ID
                # nb - if you change something like vw and want to return you have to change the run ID otherwise it will load previous results. 
                if (self.theta.keys() == saved_dict.keys()):
                    print("Loading llh results from previous fit")
                    self.theta = saved_dict
                    for key in self.theta:
                        print(key,":", self.theta[key])
                    mspec = sp.log_likelihood(np.array(list(self.theta.values())),sp,2)
                    self.plot_spectrum(fname=self.ID+'_initial_fit_llh.png',**kwargs)
            else:
                print("Fitting spectrum, please wait. Perhaps grab a cup of coffee, and your result will be ready when you return :)")
                start = time.time()
                soln = minimize(sp.log_likelihood, initial_params, args=(sp,1), method='Powell',tol=1e-3) # add to args ns, z, epoly, taken from params
                mspec,_mline,_mbline,_mcont = sp.log_likelihood(soln.x,sp,2)
                self.model_spec = mspec
                self.model_line = _mline
                if Priors.params['broadlines']:
                    self.model_bline = _mbline
                self.model_cont = _mcont
                end = time.time()
                ll_time = end - start
                print("Likelihood fitting took {0:.1f} seconds".format(ll_time))

                self.theta.update( (k,x) for k,x in zip(self.theta,soln.x) )
                #self.theta['vw_fwhm_kms'] = self.theta['vw']*(2.*np.sqrt(2*np.log(2.)))
                #self.theta['vw_b_fwhm_kms'] = self.theta['vw_b']*(2.*np.sqrt(2*np.log(2.)))
                
                for key in self.theta:
                    print(key,":", self.theta[key])

                hkl.dump(self.theta, llf)

                Plotting.plot_spectrum(fname=self.ID+'_initial_fit_llh.png',**kwargs)
                
    @staticmethod
    def log_likelihood(theta, self, output):
        # its a static method because theta has to be the first arg in order for scipy minimize to work. 
        
        # initialise parameters

        broadlines = Priors.params['broadlines']

        if broadlines:
        
            npa = 4
            z, vw, vw_b, sc = theta[:npa]
        else:
            npa = 3
            z, vw, sc = theta[:npa]
            vw_b = 0.

        
        escale_coeffs = theta[npa:npa+Priors.params['epoly']]
        pscale_coeffs = theta[npa+Priors.params['epoly']:npa+Priors.params['epoly']+Priors.params['ppoly']]
        line_coeffs = theta[npa+Priors.params['epoly']+Priors.params['ppoly']:npa+Priors.params['epoly']+Priors.params['ppoly']+Priors.params['nlines']] # linecoeffs
        bline_coeffs = theta[npa+Priors.params['epoly']+Priors.params['ppoly']+Priors.params['nlines']:npa+Priors.params['epoly']+Priors.params['ppoly']+Priors.params['nlines']+Priors.params['nblines']] # linecoeffs
        cont_coeffs = theta[npa+Priors.params['epoly']+Priors.params['ppoly']+Priors.params['nlines']+Priors.params['nblines']:] # the rest are nspline coeffs 
        
        coeffs = theta[npa+Priors.params['epoly']+Priors.params['ppoly']:] #line and spline coeffs. 

        #print(len(coeffs))
      
        #print(pscale_coeffs)
        # make model for continuum and lines


        templ_arr,tline = Models.generate_templates(z,sc,vw,vw_b,init=False)

        

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
            
        xr = Data.spec_wobs
        xr_ang = xr*1e4

        
        mask = Data.valid
        flam = Data.spec_fnu*Data.to_flam
        eflam = Data.spec_efnu*Data.to_flam

        
        
        escale = (np.polyval(escale_coeffs,xr))
        #print(np.nanmedian(escale))
        e2 = (eflam*escale)**2. # scale errors in fnu, THEN convert to flam??



        if Priors.params['scale_p']==False:
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
        # prior: broad lines cannot be negative coeffs. narrow lines can. 
        if broadlines:
            logprior = Priors.z_rv.logpdf(z) + Priors.escale_prior(escale) + Priors.sc_prior(sc) + Priors.vw_prior(vw) + Priors.vwb_prior(vw_b) + Priors.coeffs_prior(coeffs) #+self.balmer_ratios_prior(line_fluxes)
        else:
            logprior = Priors.z_rv.logpdf(z) + Priors.escale_prior(escale) + Priors.sc_prior(sc) + Priors.vw_prior(vw)
        lprob = lnp + logprior + lnphot

        if np.isnan(lprob):
            lprob = -np.inf
        else:
            lprob = lprob
    
      
        if output==0:
            #log-likelihood 
            return lprob
        elif output==1:
            #print(-2.*lprob)
            return -2.*lprob # chi2
        else:
            return mspec*pscale,_mline*pscale,_mbline*pscale,_mcont*pscale
        
