import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_MAIN_FREE"] = "1"

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
    
    def __init__(self):
      
      # bla bla bla 
      self.fit_redshift()
      # 
    
       
    def fit_redshift(self,zgrid,zstep,sc,vw,vw_b,zfix=None): 
        
        mask = valid
        flam = spec_fnu*to_flam
        eflam = spec_efnu*to_flam


        flam[~mask] = np.nan
        eflam[~mask] = np.nan
        
        
        if isinstance(zfix, float): 
            
            _A,line_names_,broad_line_names_,tline = self.generate_templates(zfix,sc,vw,vw_b,init=True)
        
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
            #cv_mask = list(np.ones(self.params['nspline'],dtype='bool')).append(coeffs[:self.params['nlines']]!=0)
            # spl_list = list(np.ones(self.params['nspline'],dtype='bool'))
            # cv_mask0 = list(coeffs[:self.params['nlines']+self.params['nblines']]!=0)
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

                _A,line_names_,broad_line_names_,tline = self.generate_templates(z,sc,vw,vw_b,init=True)

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
        
    def init_params(self,fix_ns=True,ns=13,epoly=4,ppoly=2,vw=100.,vw_b=300.,sc=1.3,z=None,z0=[3.,4.],zstep=None,halpha_prism='free',scale_p=False,broadlines=False):
        # initialise input params for log-likelihood fitting
           # could also include initial redshift fit? or we put that somewhere else?"""
        # return self.params which has params either fixed or variable 
        # if fixed, it has the option of which ones to fix and at which values 

        self.params = {}

        self.params['sc']=sc
        self.params['vw_prior'] = vw
        self.params['vwb_prior'] = vw_b
        self.params['fix_ns']=fix_ns
        self.params['nspline']=ns
        self.params['epoly']=epoly
        self.params['scale_p']=scale_p
        if self.params['scale_p']==True:
            self.params['ppoly'] = ppoly
        else:
            self.params['ppoly'] = 0
            
        self.params['halpha_prism']=halpha_prism
        self.params['broadlines']=broadlines
        
        

        escale_coeffs = np.ones(epoly) #npoly 
        
        if self.params['scale_p']==True:
            pscale_coeffs = np.ones(ppoly) # np.ones(ppoly)
        else:
            pscale_coeffs = None
            
        self.initialize_bsplines() #makes continuum splines
        
        
        is_prism = self.grating in ['prism']

        # FIT REDSHIFT 

            # if input z is given, use that, otherwise use redshift grid 
        if z is not None:
            print("fix redshift", z)
            self.params['zbest']=z
            zbest = z 
            zgrid = 1
            zstep = 1
            _A, coeffs, covard, line_names_,broad_line_names_,tline = self.__fit_redshift(zgrid,zstep,sc,vw,vw_b,zfix=z)

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
    
            zgrid = utils.log_zgrid(z0, step0)
            
            zg0, chi0 = self.__fit_redshift(zgrid,zstep,sc,vw,vw_b,zfix=None)
            
            zbest0 = zg0[np.argmin(chi0)]
            
            # Second pass
            zgrid1 = utils.log_zgrid(zbest0 + np.array([-0.005, 0.005])*(1+zbest0), 
                                step1)
            
            zg1, chi1 = self.__fit_redshift(zgrid1,zstep,sc,vw,vw_b,zfix=None)
            
            zbest = zg1[np.argmin(chi1)]
            
            self.params['zbest']=zbest
        
            _A, coeffs, covard, line_names_, broad_line_names_,tline = self.__fit_redshift(zgrid1,zstep,sc,vw,vw_b,zfix=zbest)
        
        self.templates = _A
        #self.tline = tline

        nline_mask = tline==0
        nbline_mask = tline==1
        

        _model = _A.T.dot(coeffs)
        _mline = _A.T.dot(coeffs*nline_mask)
        if broadlines:
            _mbline = _A.T.dot(coeffs*nbline_mask)        
            _mcont = _model - _mline - _mbline
            self.model_bline = _mbline
        else:
            _mcont = _model - _mline
        
        self.model_spec = _model
        self.model_line = _mline
        self.model_cont = _mcont

        mask = valid
        flam = spec_fnu*to_flam
        eflam = spec_efnu*to_flam


        flam[~mask] = np.nan
        eflam[~mask] = np.nan

        full_chi2 = ((flam - _model)**2./eflam**2.)[mask].sum()
        cont_chi2 = ((flam - _mcont)**2./eflam**2.)[mask].sum()

        print(f'full chi2 = {full_chi2}, cont chi2 = {cont_chi2}')

        _Acont = (_A.T*coeffs)[mask,:][:,self.params['nlines']+self.params['nblines']:]
        _Acont[_Acont < 0.001*_Acont.max()] = np.nan

        self.Acont = _Acont

        
        line_coeffs = coeffs[:self.params['nlines']]
        cont_coeffs = coeffs[self.params['nlines']+self.params['nblines']:]
        line_covard = covard[:self.params['nlines']]
        cont_covard = covard[self.params['nlines']+self.params['nblines']:]   

        if broadlines:

            bline_coeffs = coeffs[self.params['nlines']:self.params['nlines']+self.params['nblines']]
            bline_covard = covard[self.params['nlines']:self.params['nlines']+self.params['nblines']]
        
        theta_dict = OrderedDict()
        
        theta_dict['z'] = zbest
        theta_dict['vw'] = vw
        if broadlines:
            theta_dict['vw_b'] = vw_b
        theta_dict['sc']=sc
        
        escale_names = [f'escale_{x}' for x in range(epoly)]
        for name,coeff in zip(escale_names,escale_coeffs):
            theta_dict[name]=coeff

            
        if self.params['scale_p']==True:
            pscale_names = [f'pscale_{x}' for x in range(ppoly)]
            for name,coeff in zip(pscale_names,pscale_coeffs):
                theta_dict[name]=coeff

        
        bspl_names = [f'bspl_{x}' for x in range(len(cont_coeffs))]
        
         # remove line coeffs that are zero 
        
        line_mask = line_coeffs!=0. 
        self.params['nlines']=np.sum(line_mask)


        if broadlines:
            bline_mask = bline_coeffs!=0. 
            self.params['nblines']=np.sum(line_mask)


        spl_mask = cont_coeffs!=0. 
        self.params['nspline']=np.sum(spl_mask)
        self.spl_mask = spl_mask

        for name,coeff in zip(np.array(line_names_)[line_mask].tolist(),line_coeffs[line_mask]):
            theta_dict['line '+name]=coeff

        if broadlines:

            for name,coeff in zip(np.array(broad_line_names_)[bline_mask].tolist(),bline_coeffs[bline_mask]):
                theta_dict['broad line '+name]=coeff
            
        for name,coeff in zip(np.array(bspl_names)[spl_mask].tolist(),cont_coeffs[spl_mask]):
            theta_dict[name]=coeff
                  
        self.theta = theta_dict

        line_tab = OrderedDict()
        

        for name,coeff,covard in zip(np.array(line_names_)[line_mask].tolist(),line_coeffs[line_mask],line_covard[line_mask]):
            line_tab['line '+name]=[coeff,covard]

        self.line_table = line_tab

        if broadlines:

            bline_tab = OrderedDict()
            
            for name,coeff,covard in zip(np.array(broad_line_names_)[bline_mask].tolist(),bline_coeffs[bline_mask],bline_covard[bline_mask]):
                bline_tab['broad line '+name]=[coeff,covard]
    
            self.bline_table = bline_tab
            
    def fit_spectrum(self,fix_ns=True,ns=13,epoly=4,ppoly=3,vw=100.,vw_b=300.,sc=1.3,z=3.652,z0=[3.,4.],zstep=None,halpha_prism='free',ll=False,scale_p=False,broadlines=False,**kwargs):
        """ initialises parameters and minimises -log likelihood """
            
        from scipy.optimize import minimize
        np.random.seed(42)

        self.init_params(fix_ns,ns,epoly,ppoly,vw,vw_b,sc,z,z0,zstep,halpha_prism,scale_p,broadlines)
        for key in self.theta:
            print(key,":", self.theta[key])
        initial_params = np.array(list(self.theta.values()))
        
        self.plot_spectrum(fname=self.ID+"initial_fit.png",**kwargs)
        
        
        
        if ll:

            self.init_logprior()
            llf = self.ID+'_llh_theta.dict'
            print(llf)
            
            # check likelihood fitting hasn't already been run, saves a bit of time 
            if os.path.isfile(llf): 
                saved_dict = hickle.load(llf)
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
                if self.params['broadlines']:
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

                hickle.dump(self.theta, llf)

                self.plot_spectrum(fname=self.ID+'_initial_fit_llh.png',**kwargs)
                
    @staticmethod
    def log_likelihood(theta, self, output):
        # its a static method because theta has to be the first arg in order for scipy minimize to work. 
        
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
        # prior: broad lines cannot be negative coeffs. narrow lines can. 
        if broadlines:
            logprior = self.z_rv.logpdf(z) + self.escale_prior(escale) + self.sc_prior(sc) + self.vw_prior(vw) + self.vwb_prior(vw_b) + self.coeffs_prior(coeffs) #+self.balmer_ratios_prior(line_fluxes)
        else:
            logprior = self.z_rv.logpdf(z) + self.escale_prior(escale) + self.sc_prior(sc) + self.vw_prior(vw)
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
        
