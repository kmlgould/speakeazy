import os
import re
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import hickle as hkl
from tqdm import tqdm

from astropy.table import Table
import astropy.units as u
from astropy.io import fits

import eazy
from grizli import utils
from scipy.optimize import lsq_linear

utils.set_warnings()

from pylab import *

from .model_spectra import Model
from .priors import Priors
from .data import Data

rc('axes', linewidth=1.5)
rc('xtick',direction='in')#, minor_visible=True, major_width=1.2, minor_width=1.0)
rc('ytick',direction='in')#, minor_visible=True, major_width=1.2, minor_width=1.0)
rc('font',size=14)
plt.rcParams["font.family"] = "serif"

class Fitter(object):
    """Fitter

    Class for fitting spectral data with models. Performs simple nnls chi-squared fitting a la eazy, or 
    interaces with EMCEE to do priors based sampling. 
    

    Arguments:
        data -- speakeazy.data 
                A data object containing spectral data / photometric data to be fit. 
                
        priors -- speakeazy.priors
                Prior object containing fit instructions and priors for fitting. 
                
    Attributes:
    
    
    
    Functions:
        __fit_redshift -- 
        
        init_params -- 
        
        fit_spectrum -- 
        
        log_likelihood -- 
    """
    
    def __init__(self,data,priors):
        
        #check priors have been initialized, if not, set them here 
        
        if hasattr(priors,'params'):
            self.priors = priors
        else:
            self.priors = Priors(data)
            
         #create model instance which now has data info, prior info and has created splines.
        
        self.model = Model(data=data,priors=priors) 
        
        self.data = data
        
        # check there are params in the priors class instance here 
        
       # if hasattr(priors,'params'):
       #     self.priors = priors
       #     self.params = self.priors.params
       # else:
       #     self.priors = Priors(self.data)
       #     self.params = self.priors.set_params(self)
            
      #  self.models = models
        
    
       
    def fit_redshift_grid(self,zgrid,scale_disp=1.3,vel_width=100.,vel_width_broad=1000.,zfix=None): 
        """__fit_redshift _summary_

        _extended_summary_

        Arguments:
            zgrid -- _description_
            sc -- _description_
            vw -- _description_
            vw_b -- _description_

        Keyword Arguments:
            zfix -- _description_ (default: {None})

        Returns:
            _description_
        """
        
        mask = self.data.valid
        flam = self.data.spec_fnu*self.data.to_flam
        eflam = self.data.spec_efnu*self.data.to_flam


        flam[~mask] = np.nan
        eflam[~mask] = np.nan
        
        if isinstance(zfix, float): 
            
            _A,line_names_,broad_line_names_,tline = self.model.generate_templates(self.data,zfix,scale_disp,vel_width,vel_width_broad,theta=None,chisq=True)
        
            okt = _A[:,mask].sum(axis=1) > 0
            _Ax = _A[okt,:]/eflam
            _yx = flam/eflam
            #_x = np.linalg.lstsq(_Ax[:,mask].T, 
            #                         _yx[mask], rcond=None)
            
            fit_bounds = self.model.fit_bounds
            masked_fit_bounds = (fit_bounds[0][:][okt] ,fit_bounds[1][:][okt])
            res = lsq_linear(_Ax[:,mask].T, 
                                     _yx[mask],
                                     bounds=masked_fit_bounds, method='bvls',verbose=True)
            #_x = nnls(_Ax[:,mask].T, 
            #                     _yx[mask])

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

                _A,line_names_,broad_line_names_,tline = self.model.generate_templates(self.data,z,scale_disp,vel_width,vel_width_broad,theta=None,chisq=True)

                okt = _A[:,mask].sum(axis=1) > 0
                _Ax = _A[okt,:]/eflam
                _yx = flam/eflam
                #_x = np.linalg.lstsq(_Ax[:,mask].T, 
                #                         _yx[mask], rcond=None)
                fit_bounds = self.model.fit_bounds
                masked_fit_bounds = (fit_bounds[0][:][okt] ,fit_bounds[1][:][okt])
                res = lsq_linear(_Ax[:,mask].T, 
                                     _yx[mask],
                                     bounds=masked_fit_bounds, method='bvls',verbose=True)
                _x = res[0]
                #_x = nnls(_Ax[:,mask].T, 
                #                 _yx[mask])


                coeffs = np.zeros(_A.shape[0])
                coeffs[okt] = _x[0]
                _model = _A.T.dot(coeffs)

                chi = (flam - _model) / eflam

                chi2_i = (chi[mask]**2).sum()
                chi2[iz] = chi2_i
        
            return zgrid, chi2
        
    def fit_redshift_chisq(self,zstep=[0.002,0.0001]):
        """fit_redshift_chisq _summary_

        _extended_summary_
        """       

        self.theta = None
        self.line_table = None
        self.bline_table = None 
        #self.params = Priors.set_params(self,**input_params)

        escale_coeffs = np.ones(self.priors.params['epoly']) #npoly 
        
        if self.priors.params['scale_p']==True:
            pscale_coeffs = np.ones(self.priors.params['ppoly']) # np.ones(ppoly)
        else:
            pscale_coeffs = None
            
    
        
        is_prism = self.data.grating in ['prism']

        # FIT REDSHIFT 

            # if input z is given, use that, otherwise use redshift grid 
        if self.priors.params['z_in'] is not None:
            print("fix redshift", self.priors.params['z_in'])
            self.priors.params['zbest']=self.priors.params['z_in']
            zbest = self.priors.params['z_in']
            zgrid = 1
            zstep = 1
            _A, coeffs, covard, line_names_,broad_line_names_,tline = self.fit_redshift_grid(zgrid,self.priors.params['scale_disp'],self.priors.params['vel_width'],self.priors.params['vel_width_broad'],zfix=self.priors.params['z_in'])

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
    
            zgrid = utils.log_zgrid(self.priors.params['z_range'], step0)
            
            zg0, chi0 = self.fit_redshift_grid(zgrid,self.priors.params['scale_disp'],self.priors.params['vel_width'],self.priors.params['vel_width_broad'],zfix=None)
            
            zbest0 = zg0[np.argmin(chi0)]
            
            # Second pass
            zgrid1 = utils.log_zgrid(zbest0 + np.array([-0.005, 0.005])*(1+zbest0), 
                                step1)
            
            zg1, chi1 = self.fit_redshift_grid(zgrid1,self.priors.params['scale_disp'],self.priors.params['vel_width'],self.priors.params['vel_width_broad'],zfix=None)
            
            zbest = zg1[np.argmin(chi1)]
            
            self.priors.params['zbest']=zbest
        
            _A, coeffs, covard, line_names_, broad_line_names_,tline = self.fit_redshift_grid(zgrid1,self.priors.params['scale_disp'],self.priors.params['vel_width'],self.priors.params['vel_width_broad'],zfix=zbest)
        
        self.templates = _A
        #self.tline = tline

        nline_mask = tline==0
        nbline_mask = tline==1
        

        _model = _A.T.dot(coeffs)
        _mline = _A.T.dot(coeffs*nline_mask)
        _mline_arr = _A[nline_mask,:]
        
        if self.priors.params['broadlines']:
            _mbline = _A.T.dot(coeffs*nbline_mask)        
            _mcont = _model - _mline - _mbline
            self.model_bline = _mbline
        else:
            _mcont = _model - _mline
        
        self.model_spec = _model
        self.model_line = _mline
        self.model_cont = _mcont

        mask = self.data.valid
        flam = self.data.spec_fnu*self.data.to_flam
        eflam = self.data.spec_efnu*self.data.to_flam


        flam[~mask] = np.nan
        eflam[~mask] = np.nan

        full_chi2 = ((flam - _model)**2./eflam**2.)[mask].sum()
        cont_chi2 = ((flam - _mcont)**2./eflam**2.)[mask].sum()

        print(f'full chi2 = {full_chi2}, cont chi2 = {cont_chi2}')

        _Acont = (_A.T*coeffs)[mask,:][:,self.model.nlines+self.model.nblines:]
        _Acont[_Acont < 0.001*_Acont.max()] = np.nan
        
        _Aline = (_A.T*coeffs)[mask,:][:,:self.model.nlines:]
        _Aline[_Aline < 0.0001*_Aline.max()] = np.nan
        
        if self.priors.params['broadlines']:
            _Abline = (_A.T*coeffs)[mask,:][:,self.model.nlines:self.model.nlines+self.model.nblines]
            _Abline[_Abline < 0.001*_Abline.max()] = np.nan
            self.Abline = _Abline 

        self.Acont = _Acont
        self.Aline = _Aline

        
        line_coeffs = coeffs[:self.model.nlines]
        cont_coeffs = coeffs[self.model.nlines+self.model.nblines:]
        line_covard = covard[:self.model.nlines]
        cont_covard = covard[self.model.nlines+self.model.nblines:]   

        if self.priors.params['broadlines']:

            bline_coeffs = coeffs[self.model.nlines:self.model.nlines+self.model.nblines]
            bline_covard = covard[self.model.nlines:self.model.nlines+self.model.nblines]
    
        ##### MAKE THETA DICTIONARY HERE    
        
        
        theta_dict = OrderedDict()
        
        theta_dict['z'] = zbest
        theta_dict['vel_width'] = self.priors.params['vel_width']
        if self.priors.params['broadlines']:
            theta_dict['vel_width_broad'] = self.priors.params['vel_width_broad']
        theta_dict['scale_disp']=self.priors.params['scale_disp']
        
        escale_names = [f'escale_{x}' for x in range(self.priors.params['epoly'])]
        for name,coeff in zip(escale_names,escale_coeffs):
            theta_dict[name]=coeff

            
        if self.priors.params['scale_p']==True:
            pscale_names = [f'pscale_{x}' for x in range(self.priors.params['ppoly'])]
            for name,coeff in zip(pscale_names,pscale_coeffs):
                theta_dict[name]=coeff

        
        bspl_names = [f'bspl_{x}' for x in range(len(cont_coeffs))]
        
         # remove line coeffs that are zero 
        
        line_mask = line_coeffs!=0. 
        self.model.nlines=np.sum(line_mask)


        if self.priors.params['broadlines']:
            bline_mask = bline_coeffs!=0. 
            self.model.nblines=np.sum(line_mask)


        spl_mask = cont_coeffs!=0. 
        self.priors.params['nspline']=np.sum(spl_mask)
        self.model.nspline=np.sum(spl_mask)
        self.spl_mask = spl_mask

        for name,coeff in zip(np.array(line_names_)[line_mask].tolist(),line_coeffs[line_mask]):
            theta_dict['line '+name]=coeff

        if self.priors.params['broadlines']:

            for name,coeff in zip(np.array(broad_line_names_)[bline_mask].tolist(),bline_coeffs[bline_mask]):
                theta_dict['broad line '+name]=coeff
            
        for name,coeff in zip(np.array(bspl_names)[spl_mask].tolist(),cont_coeffs[spl_mask]):
            theta_dict[name]=coeff
                  
        self.theta = theta_dict

        line_tab = OrderedDict()
        

        for name,coeff,covard in zip(np.array(line_names_)[line_mask].tolist(),line_coeffs[line_mask],line_covard[line_mask]):
            line_tab['line '+name]=[coeff,covard]

        self.line_table = line_tab

        if self.priors.params['broadlines']:

            bline_tab = OrderedDict()
            
            for name,coeff,covard in zip(np.array(broad_line_names_)[bline_mask].tolist(),bline_coeffs[bline_mask],bline_covard[bline_mask]):
                bline_tab['broad line '+name]=[coeff,covard]
    
            self.bline_table = bline_tab
            
        for key in self.theta:
            print(key,":", self.theta[key])
        #initial_params = np.array(list(self.theta.values()))
        
        return
      
    """  
    def fit_spectrum(self,ll=False,**kwargs):
      
            
        from scipy.optimize import minimize
        np.random.seed(42)

        self.init_params()
        for key in self.theta:
            print(key,":", self.theta[key])
        initial_params = np.array(list(self.theta.values()))
        
        self.plot_spectrum(fname=self.ID+"initial_fit.png",**kwargs)
        
        
        
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
                if self.priors.params['broadlines']:
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

                self.plot_spectrum(fname=self.ID+'_initial_fit_llh.png',**kwargs)
    """    
    @staticmethod
    def log_likelihood(theta, self, output):
        # its a static method because theta has to be the first arg in order for scipy minimize to work. 
        
        # initialise parameters

        broadlines = self.priors.params['broadlines']

        if broadlines:
        
            npa = 4
            z, vw, vw_b, sc = theta[:npa]
        else:
            npa = 3
            z, vw, sc = theta[:npa]
            vw_b = 0.

        
        escale_coeffs = theta[npa:npa+self.priors.params['epoly']]
        pscale_coeffs = theta[npa+self.priors.params['epoly']:npa+self.priors.params['epoly']+self.priors.params['ppoly']]
        line_coeffs = theta[npa+self.priors.params['epoly']+self.priors.params['ppoly']:npa+self.priors.params['epoly']+self.priors.params['ppoly']+self.model.nlines] # linecoeffs
        bline_coeffs = theta[npa+self.priors.params['epoly']+self.priors.params['ppoly']+self.model.nlines:npa+self.priors.params['epoly']+self.priors.params['ppoly']+self.model.nlines+self.model.nblines] # linecoeffs
        cont_coeffs = theta[npa+self.priors.params['epoly']+self.priors.params['ppoly']+self.model.nlines+self.model.nblines:] # the rest are nspline coeffs 
        
        coeffs = theta[npa+self.priors.params['epoly']+self.priors.params['ppoly']:] #line and spline coeffs. 

        #print(len(coeffs))
      
        #print(pscale_coeffs)
        # make model for continuum and lines


        templ_arr,tline = self.models.generate_templates(z,sc,vw,vw_b,theta=self.theta,init=False)

        

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



        if self.priors.params['scale_p']==False:
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
   
        
    def plot_spectrum(self,save=False,fname=None,flat_samples=None,line_snr=5.,show_lines=False,ylims=None,xlims=None):
        
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
        
        if (self.priors.params['scale_p']==True):
            if (self.theta['pscale_0']!=1.) : 
                pscale_coeffs = np.array([self.theta[f'pscale_{i}'] for i in range(self.priors.params['ppoly'])])
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

        if hasattr(self, 'model_spec'):
            #if (self.params['scale_p']==True):
            #    if (self.theta['pscale_0']!=1.):
                    # scaling has already been applied 
          #          plt.step(wav[mask],(self.model_spec)[mask],color='blue',label='Model')
        #else:
            plt.step(wav[mask],(self.model_spec*scale)[mask],color='black',label='Model')

            
            plt.step(wav[mask],(self.model_line*scale)[mask],color='blue',label='Lines')
            #plt.step(wav[mask],(self.model_bline*scale)[mask],color='red',label='Broad lines')
            #plt.step(wav[mask],self.model_cont[mask],color='olive',label='Continuum')

            if hasattr(scale, "__len__"):
                plt.plot(self.data.spec_wobs[mask], (((self.Acont.T).T))*scale[mask,None],
                        color='olive', alpha=0.3)
                plt.plot(self.data.spec_wobs[mask], (((self.Aline.T).T))*scale[mask,None],
                        color='blue', alpha=0.3)
            else:
                plt.plot(self.data.spec_wobs[mask], (((self.Acont.T).T))*scale,
                        color='olive', alpha=0.3)
                plt.plot(self.data.spec_wobs[mask], (((self.Aline.T).T))*scale,
                        color='blue', alpha=0.3)
                
            # plot emission lines 

            if show_lines:

                
                
                for line in self.line_table:
                    l_snr = abs(self.line_table[line][0])/abs(self.line_table[line][1])
                    if l_snr>line_snr:
                        #lname = line.strip(' line') # get name of line
                        lname = re.sub(r'line ', '', line)
                        if len(self.data.lw[lname])>0:
                            wavl = np.average(self.data.lw[lname])
                        else:
                            wavl = self.data.lw[lname]
                        line_center = (wavl*(1.+self.theta['z']))/1e4
                        if (xlims[0]<line_center<xlims[1]):
                            plt.axvline(line_center,ls='dashed',color='blue',alpha=0.5)
                            plt.text(x=line_center,y=ymax*0.5,s=lname,rotation=90,fontsize=9,color='blue',alpha=0.5)
                        else:
                            continue
                """
                for bline in self.bline_table:
                    l_snr = abs(self.bline_table[bline][0])/abs(self.bline_table[bline][1])
                    if l_snr>line_snr:
                        lname = re.sub(r'broad line ', '', bline) 
                        line_center = (self.lw[lname][0]*(1.+self.theta['z']))/1e4
                        if (xlims[0]<line_center<xlims[1]):                        
                            plt.axvline(line_center,ls='dashed',color='red',alpha=0.5)
                            plt.text(x=line_center,y=ymax*0.5,s=lname,rotation=90,fontsize=9,color='red',alpha=0.5)
                        else:
                            continue
                """



            
        #if flat_samples is not None:
        #    for sample in flat_samples:
         #       mspec = sp.log_likelihood(sample,sp,2)
        #        plt.plot(wav,mspec,color='cornflowerblue',lw=0.5,alpha=0.3)
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel(r'F$_{\lambda}$ [10$^{-19}$ erg/s/cm$^{2}/\AA$]')
        zb = self.theta['z']
        plt.title(f'{self.data.fname}, zspec={zb:.3f}',fontsize=10)
        plt.text(x=0.6,y=0.8,s=f'{self.data.fname}\n z={zb:.3f}',
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
            plt.savefig(fname)
        return #fig