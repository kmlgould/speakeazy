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

spec_wobs = None
valid = None
spec_fnu = None
spec_efnu = None
to_flam = None
spec_R_fwhm = None

class Spectrum(object):
    
    """ Main object for fitting spectra using emcee. Loads data, initialises templates, priors, 
    peforms fitting, loads and plots results. 
    
    #ToDo: add photometry - first try is without. 
    
    Parameters (stuff you give it to make it - like ingredients) 
    ----------
    
    spectrum_file: str
        filename of 1D spectrum in format xx 
    
    # ToAdd
    params: dict 
        Dictionary containing params (spectrum filename, 
        photometry filename, translate file, zeropoint) .
    
    translate_file: str
        Translation filename for `eazy.param.TranslateFile`.
        
    Attributes (stuff it has) 
    ----------
    
    
    
    
    Methods (stuff it does) 
    ----------
    """
    
    def __init__(self,spectrum_file,photometry_file,run_ID,phot_id,xwave=None):

        reload(msaexp.resample_numba); reload(msaexp.spectrum)
        reload(msaexp.resample_numba); reload(msaexp.spectrum)

        
        try:
            from msaexp.resample_numba import \
                resample_template_numba as resample_func
        except ImportError:
            from .resample import resample_template as resample_func
            
        self.resample_func = resample_func
        self.spectrum_file = spectrum_file
        self.photometry_file = photometry_file
        self.run_ID = run_ID
        self.phot_id = phot_id

        if xwave is not None:
            self.xwave = xwave
        else:
            self.xwave = None
        
        here =  os.getcwd()
        
        newpath = here+f"/{run_ID}/"
        
        if not os.path.exists(newpath):
            print(f"Creating new folder for outputs in {newpath}")
            os.makedirs(newpath)
        
        if 'spec.fits' in spectrum_file:
            froot = spectrum_file.split('.spec.fits')[0]
        else:
            froot = spectrum_file.split('.fits')[0]
            
        self.fname = froot
                
        self.ID = newpath+froot
        #spec_wobs = None
        
        self.initialize_spec()

        if self.photometry_file:
            self.initialize_phot()
        else:
            print("No photometry found")

        self.initialize_emission_line()

    def initialize_phot(self):
    
        #eazy.symlink_eazy_inputs()
        res = eazy.filters.FilterFile()
    
        phot = Table.read('sep_bcg_subtracted_triple.fits')
        
        phot['id'] = [286,234,458] # currently using sep photometry but need to change to photutils
    
        sel = phot['id'] == self.phot_id
    
        phot_id = phot[sel]
        
        filts = ['F090W','F115W','F200W','F277W','F356W','F410M','F444W']
        
        phot_names = [f'FLUX_AUTO_{f}' for f in filts]
        phote_names = [f'FLUXERR_AUTO_{f}' for f in filts]
    
        phot_orig = np.lib.recfunctions.structured_to_unstructured(np.array((phot_id[phot_names])))
        photerr_orig = np.lib.recfunctions.structured_to_unstructured(np.array((phot_id[phote_names])))
        
        phot_uJy = phot_orig*1e-3 #convert from nJy to uJy
        phote_uJy = photerr_orig*1e-3 #convert from nJy to uJy
    
        self.phot_flam = self.ujy_to_flam(phot_uJy,\
                                     np.array([9022.922,11543.009,19886.479,27577.959,35682.277,40820.574,44036.71]))
        self.phote_flam = self.ujy_to_flam(phote_uJy,\
                                     np.array([9022.922,11543.009,19886.479,27577.959,35682.277,40820.574,44036.71])) 

        print(self.phot_flam)
        print(self.phote_flam)
    
        # missing: f105w, 125w, f160w, didn't include hst. 
        
        eazy_filts = []
        
        for f in filts: 
            fi = f.lower()
            t = res.search(f'nircam_{fi}', case=False, verbose=True)
            if len(t)==1:
                filt = res[t[0]+1] # ...bug? 
                eazy_filts.append(filt)
    
        self.filters = eazy_filts
        
        
    def initialize_spec(self):
        """
        Read in 1D spectrum file and process key properties
        Currently assumes output is from msaexp, in fits format with an extension named SPEC1D

        err_mask : float, float or None
                Mask pixels where ``err < np.percentile(err[err > 0], err_mask[0])*err_mask[1]``
    
        err_median_filter : int, float or None
                Mask pixels where ``err < nd.median_filter(err, err_median_filter[0])*err_median_filter[1]``
        """
        import scipy.ndimage as nd
        err_mask=(10,0.5)
        err_median_filter=[11, 0.8]
        # open file 
        with fits.open(self.spectrum_file) as im:
            if 'SPEC1D' in im:
                print('spec1d found')
                spec = utils.read_catalog(im['SPEC1D'])
                
                grating = spec.meta['GRATING'].lower()
                _filter = spec.meta['FILTER'].lower()

                print(f"Reading in spectrum with grating {grating}/{_filter} combination")
                
                # replace masks with zeros and avoid nans and zeros
                for c in ['flux','err']:
                    if hasattr(spec[c], 'filled'):
                        spec[c] = spec[c].filled(0)
        
                vvalid = np.isfinite(spec['flux']+spec['err']) # shouldn't this be / not + ? 
                vvalid &= spec['err'] > 0
                vvalid &= spec['flux'] != 0 
                vvalid &= abs(spec['flux'])<1e10 # in case of fucky edges of grism... 
                vvalid &= abs(spec['err'])<1e10 # in case of fucky edges of grism... 

                # for particular cases

                if self.xwave is not None:

                    vvalid &= (spec['wave']>self.xwave[0]) & (spec['wave']<self.xwave[1])


                # for medium and high res, there are slit gaps. Let's take care of those. 

                ###############################################################################

                # edited 31/7/2023 - i think this was entirely a waste of time ... ffs 

                """
                print(len(spec['wave']))
                wav = spec['wave'].value.astype(np.float32)

                plt.plot(wav,spec['flux'])
                plt.plot(wav[valid],spec['flux'][valid])
                plt.xlim(3.6,3.8)
                plt.show()
                print(len(wav))

                testsel = (spec['wave']>3.64) & (spec['wave']<3.7)
                
                if grating != "prism":
                    gap_arr = np.array([(wav[i+1]-wav[i]) for i in range(len(wav)-1)])
                    print(gap_arr)
                    wr = np.array([i for i in range(len(wav)-1)])
                    print(wr)
                    print(np.nanmax(gap_arr))
                    plt.figure()
                    plt.scatter(wr,gap_arr)
                    plt.show()
                    mask_gap = gap_arr>0.02 #this should work for all gaps = gap is ~0.04,0.07,0.1 micron depending on grating/filter combination
                    idx_gap = np.where(mask_gap==True)[0][0] #find where gap starts
                    mask_gap[idx_gap-5:idx_gap+5]=True # mask the elements either side of the gap - should possibly be different? 
                    slit_gap_mask = np.invert(np.append(mask_gap,False)) # invert to get boolean where there is no gap, and make sure its the same length as original 

                    print(len(slit_gap_mask))
                    #valid &= slit_gap_mask # add to valid 
                """

                ###############################################################################

                if (vvalid.sum() > 0) & (err_mask is not None):
                    _min_err = np.nanpercentile(spec['err'][vvalid], err_mask[0])*err_mask[1]
                    vvalid &= spec['err'] > _min_err
        
                if err_median_filter is not None:
                    med = nd.median_filter(spec['err'][vvalid], err_median_filter[0])
                    medi = np.interp(spec['wave'], spec['wave'][vvalid], med, left=0, right=0)
                    vvalid &= spec['err'] > err_median_filter[1]*medi
        
                spec['flux'][~vvalid] = 0.
                spec['err'][~vvalid] = 0.
                spec['valid'] = vvalid
                
                um = spec['wave'].unit
                if um is None:
                    um = u.micron

        
                _data_path = "/home/ec2-user/msa_nirspec_disp_curves" #make this generalised (in msaexp data dir) 
                disp = utils.read_catalog(f'{_data_path}/jwst_nirspec_{grating}_disp.fits')


                #spec['R'] = np.interp(spec['wave'], disp['WAVELENGTH'], disp['R'],
                #                      left=disp['R'][0], right=disp['R'][-1])
                
                flam_unit = 1.e-19*u.erg/u.second/u.cm**2/u.Angstrom # change to attribute 
                self.equiv = u.spectral_density(spec['wave'].data*um)
                global to_flam
                to_flam = (1.*spec['flux'].unit).to(flam_unit, equivalencies=self.equiv).value #property 
                self.flamunit = flam_unit.unit

                global spec_R_fwhm
                global valid
                global spec_wobs
                global spec_fnu
                global spec_efnu
                spec_R_fwhm = (np.interp(spec['wave'], disp['WAVELENGTH'], disp['R'],
                                      left=disp['R'][0], right=disp['R'][-1])).astype(np.float32)
                valid = spec['valid']
                spec_wobs = spec['wave'].value.astype(np.float32)
                spec_fnu = spec['flux'].value.astype(np.float32)
                spec_efnu = spec['err'].value.astype(np.float32)
                self.grating = grating
                self.filter = _filter


            # otherwise assume its from the ETC

            else:
                spec = im[1].data
                um = u.micron
                wave = spec['WAVELENGTH']
                fluxin = spec['extracted_flux_plus_bg_resel']
                flux = np.array([self.flam_to_ujy(fl*1e-19,ww*1e4) for (fl,ww) in zip(fluxin,wave)])
                _data_path = "/home/ec2-user/msa_nirspec_disp_curves" #make this generalised (in msaexp data dir) 
                grating = "prism"
                _filter = "clear"
                disp = utils.read_catalog(f'{_data_path}/jwst_nirspec_{grating}_disp.fits')
                flam_unit = 1.e-19*u.erg/u.second/u.cm**2/u.Angstrom # change to attribute 
                fnu_unit = 1e-3*u.jansky
                self.equiv = u.spectral_density(wave*um)
                to_flam = 1e-3*(1.*fnu_unit).to(flam_unit, equivalencies=self.equiv).value #property 
                self.flamunit = flam_unit.unit
                spec_R_fwhm = (np.interp(wave, disp['WAVELENGTH'], disp['R'],
                                      left=disp['R'][0], right=disp['R'][-1])).astype(np.float32)
                valid = np.ones(len(wave),dtype='bool')
                spec_wobs = wave.astype(np.float32)
                spec_fnu = flux.astype(np.float32)
                spec_efnu = np.random.normal(0,1,len(flux))
                self.grating = grating
                self.filter = _filter


    @property
    def NWAVE(self):
        if spec_wobs is not None:
            return len(spec_wobs)
        else:
            return 0

    @property
    def MED_ESCALE(self):
        escale_coeffs = np.array([self.theta[f'escale_{i}'] for i in range(self.params['epoly'])])
        escale = (np.polyval(escale_coeffs,spec_wobs))


        plt.figure()
        plt.plot(spec_wobs,escale)
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel('Error scaling')
        plt.show()

        return np.nanmedian(escale)
        
    @property
    def MED_PSCALE(self):
        pscale_coeffs = np.array([self.theta[f'pscale_{i}'] for i in range(self.params['ppoly'])])
        pscale = (np.polyval(pscale_coeffs,spec_wobs))


        plt.figure()
        plt.plot(spec_wobs,pscale)
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel('Photometry scaling')
        plt.show()

        return np.nanmedian(pscale)

    @staticmethod
    def ujy_to_flam(data,lam):
            flam = ((3e-5)*data)/((lam**2.)*(1e6))
            return flam
        
    @staticmethod
    def flam_to_ujy(data,lam):
            fnu = ((lam**2.)*(1e6)*data)/(3e-5)
            return fnu
    
    
    def initialize_emission_line(self, nsamp=64):
        """
        Initialize emission line
        """
        self.xline = np.linspace(-nsamp, nsamp, 2*nsamp+1)/nsamp*0.1+1
        self.yline = self.xline*0.
        self.yline[nsamp] = 1
        self.yline /= np.trapz(self.yline, self.xline)
        
        lw, lr = utils.get_line_wavelengths()
        self.lw = lw
        self.lr = lr
        
        
    def emission_line(self, line_um, line_flux=1, scale_disp=1.0, velocity_sigma=100., nsig=4):
        """
        Generate emission line 
        """
        res = self.resample_func(spec_wobs,
                                 spec_R_fwhm*scale_disp, 
                                 self.xline*line_um,
                                 self.yline,
                                 velocity_sigma=velocity_sigma,
                                 nsig=nsig)
        return res*line_flux/line_um


    def bspline_array(self, nspline=13, log=False):
        """
        Generate bspline for continuum models 
        """
        bspl = utils.bspline_templates(wave=spec_wobs*1.e4,
                                       degree=3,
                                       df=nspline,
                                       log=log,
                                       get_matrix=True
                                       )
        return bspl.T
    
    def initialize_bsplines(self):
        """ Make the bspline arrays """
        if self.params['nspline'] is not None:
            self.bsplines = self.bspline_array(nspline=self.params['nspline'],log=False) #initialise
                      
    def init_logprior(self):
        from scipy.stats import expon, gamma, norm, uniform

        if self.grating != "prism":
            zscale = 0.0001
            sc_scale = 0.01
        else:
            zscale = 0.1
            sc_scale = 0.5
            
        vw_scale = 1.
        vw_b_scale = 1.
        epscale = 0.1

        if self.params['broadlines']:
            self.prior_widths = [1e-3,1.,1.,sc_scale,epscale]
        else:
            if self.grating == "prism":
                self.prior_widths = [1e-3,1.,sc_scale,epscale]
            else:
                self.prior_widths = [1e-3,10.,sc_scale,epscale]
        self.z_rv = norm(loc=self.params['zbest'],scale=zscale)
        self.vw_rv = uniform(loc=0.,scale=1000.)
        self.vwb_rv = uniform(loc=1000.,scale=5000.)
        self.escale_rv=norm(loc=1.,scale=epscale)
        self.sc_rv = norm(loc=self.params['sc'],scale=sc_scale)
        #self.sc_rv = uniform(loc=1.,scale=1.)

        

        # Balmer line ratios for Ha,Hb,Hg,Hd, prior based on case B, ratios from Groves et al. 2011
        # https://arxiv.org/pdf/1109.2597.pdf

        hahb_lr = self.lr['Balmer 10kK'][0] 
        hahg_lr = self.lr['Balmer 10kK'][0]*(1./self.lr['Balmer 10kK'][2])
        hahd_lr = self.lr['Balmer 10kK'][0]*(1./self.lr['Balmer 10kK'][3])
        #print("2.86, 0.47, 0.26")
        #print(hahb_lr,hghb_lr,hdhb_lr)
        
        self.hahb_rv = norm(loc=hahb_lr,scale=10.)
        self.hahg_rv = norm(loc=hahg_lr,scale=10.)
        self.hahd_rv = norm(loc=hahd_lr,scale=10.)
        
    #vw prior

    def vw_prior(self,vw):
        if not (vw>0.) & (vw<1000.):
            return -np.inf 
        return self.vw_rv.logpdf(vw)

    def vwb_prior(self,vw_b):
        if not (vw_b>0.) & (vw_b>1000.):
            return -np.inf 
        return self.vwb_rv.logpdf(vw_b)


    def sc_prior(self,sc):
        if not sc>0.999:
            return -np.inf 
        return self.sc_rv.logpdf(sc)

    def coeffs_prior(self,coeffs):
        if np.any(coeffs)<0.:
            return -np.inf 
        return 0.




    def escale_prior(self,escale):
        # prior such that residuals of fit with boosted errors are normally distributed 
        #if np.any(escale)<1.:
         #   return -np.inf 
        #resid = ((flam-mspec)/(eflam*escale))[mask]
        med_escale = np.nanmedian(escale)
        return self.escale_rv.logpdf(med_escale)

    def balmer_ratios_prior(self,line_fluxes):
        
        # Hb,Hg,Hd,Ha

        Hb, Hg, Hd, Ha = line_fluxes

        #HbHg

        if ((Ha<0.) & (Hb<0.)): 
            #print('both zero')
            hahb_prior = 0. #if they're both neg we don't impose a prior 
        else: 
            hahb_prior = self.hahb_rv.logpdf(Ha/Hb)
            
        if ((Ha<0.) & (Hg<0.)):
            #print('both zero')
            hahg_prior = 0. #if they're both neg we don't impose a prior 
        else: 
            hahg_prior = self.hahg_rv.logpdf(Ha/Hg)
            
        if ((Ha<0.) & (Hd<0.)): 
            #print('both zero')
            hahd_prior = 0. #if they're both neg we don't impose a prior 
        else: 
            hahd_prior = self.hahd_rv.logpdf(Ha/Hd)
            
        return  hahb_prior + hahg_prior + hahd_prior

        

    def generate_templates(self,z,sc,vw,vw_b,init=False,broadlines=False):
        """
        Generates gaussian emission line templates.

        Parameters 
        ----------
        
        z: float
            redshift

        sc: float
            scale dispersion

        vw: float
            line velocity width km/s

        vw: float
            broad line velocity width km/s

        init: bool
            True if first run, false if used during likelihood fitting and mcmc 

        Returns
        ---------- 
        flux_arr: array [NTEMP, NWAVE] [erg/s/cm2??]
            Array of templates where NTEMP = nsplines + nlines, NWAVE = len(spectrum wavelength array)

        line_names_: list of line names, only returned if init=True 
        """

        broadlines = self.params['broadlines']
        lines = []
        # broad
        broad_lines = []
        tline = []
       # tline1 = []
   
        
        if init:
            line_names = self.get_lines(z,self.params['halpha_prism'],get_broad=False)
            line_names_ = []
            # broad
            
            if broadlines:
                broad_line_names = self.get_lines(z,self.params['halpha_prism'],get_broad=True)
                broad_line_names_ = []
            
            
        else:

            line_names1 = [l for l in self.theta.keys() if l.startswith("line ")]
       
            line_names = [re.sub(r'line ', '', l,1) for l in line_names1]
        
            # broad
            if broadlines:
                broad_line_names1 = [l for l in self.theta.keys() if l.startswith("broad line ")]
                broad_line_names = [re.sub(r'broad line ', '', l,1) for l in broad_line_names1]
   

     
            
        for k in line_names:
            if (k not in self.lw):
                continue
            else:
                line_k = np.zeros(len(spec_wobs))
                for lwk, lrk in zip(self.lw[k], self.lr[k]):
                    line_k += self.emission_line(lwk*(1.+z)/1.e4,
                                                 line_flux=lrk/np.sum(self.lr[k]),
                                                 scale_disp=sc,
                                                 velocity_sigma=vw,
                                                 nsig=4)
                lines.append(line_k)
                
                if init:
                    line_names_.append(k)
                    #tline.append(True)
                tline.append(0)

        if broadlines:
             
            for kb in broad_line_names:
                if kb not in self.lw:
                    continue
                else:
                    line_kb = np.zeros(len(spec_wobs))
                    for lwk, lrk in zip(self.lw[kb], self.lr[kb]):
                        line_kb += self.emission_line(lwk*(1.+z)/1.e4,
                                                     line_flux=lrk/np.sum(self.lr[kb]),
                                                     scale_disp=sc,
                                                     velocity_sigma=vw_b,
                                                     nsig=4)
                    # broad
                    broad_lines.append(line_kb)
                    if init:
                        broad_line_names_.append(kb)
                    tline.append(1)

        self.params['nlines'] = (np.array(lines).shape)[0]
        # broad
        if broadlines:
            self.params['nblines'] = (np.array(broad_lines).shape)[0] 
        else:

            self.params['nblines'] = 0
            
        NTEMP = self.params['nlines']+self.params['nblines']+self.params['nspline']
        flux_arr = np.zeros((NTEMP, self.NWAVE))
        
        flux_arr[0:self.params['nlines'],:] = np.array(lines)/1.e4

        if broadlines:

            flux_arr[self.params['nlines']:self.params['nlines']+self.params['nblines'],:] = np.array(broad_lines)/1.e4
        
        if init:
            flux_arr[self.params['nlines']+self.params['nblines']:,:] = self.bsplines
            for i in range(self.bsplines.shape[0]):
                tline.append(2)
        else:
            flux_arr[self.params['nlines']+self.params['nblines']:,:] = self.bsplines[self.spl_mask]
            for i in range(self.bsplines[self.spl_mask].shape[0]):
                    tline.append(2)

        if not broadlines:
            broad_line_names_ = []

        
        if init:
            return flux_arr, line_names_, broad_line_names_, np.array(tline)
        else:
            return flux_arr,np.array(tline)
        
        
    def __fit_redshift(self,zgrid,zstep,sc,vw,vw_b,zfix=None): 
        
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
        
    
    
    def plot_spectrum(self,save=True,fname=None,flat_samples=None,line_snr=5.,show_lines=False,ylims=None,xlims=None):
        mask = valid
        
        flam = spec_fnu*to_flam
        eflam = spec_efnu*to_flam
        
        flam[~mask] = np.nan
        eflam[~mask] = np.nan
        
        wav = spec_wobs
        
        xmin = np.nanmin(wav[mask])
        xmax = np.nanmax(wav[mask])

        plt.figure(figsize=(12,4))
        #plt.fill_between(wav,(flam+eflam),((flam-eflam)),\
                                # alpha=0.4,color='cornflowerblue',zorder=-99)

        scale=1.
        
        if (self.params['scale_p']==True):
            if (self.theta['pscale_0']!=1.) : 
                pscale_coeffs = np.array([self.theta[f'pscale_{i}'] for i in range(self.params['ppoly'])])
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
                plt.plot(spec_wobs[mask], (((sp.Acont.T).T))*scale[mask,None],
                        color='olive', alpha=0.3)
            else:
                plt.plot(spec_wobs[mask], (((sp.Acont.T).T))*scale,
                        color='olive', alpha=0.3)
            # plot emission lines 

            if show_lines:

                
                
                for line in self.line_table:
                    l_snr = abs(self.line_table[line][0])/abs(self.line_table[line][1])
                    if l_snr>line_snr:
                        #lname = line.strip(' line') # get name of line
                        lname = re.sub(r'line ', '', line)
                        if len(self.lw[lname])>0:
                            wavl = np.average(self.lw[lname])
                        else:
                            wavl = self.lw[lname]
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



            
        if flat_samples is not None:
            for sample in flat_samples:
                mspec = sp.log_likelihood(sample,sp,2)
                plt.plot(wav,mspec,color='cornflowerblue',lw=0.5,alpha=0.3)
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel(r'F$_{\lambda}$ [10$^{-19}$ erg/s/cm$^{2}/\AA$]')
        zb = self.theta['z']
        plt.title(f'{self.fname}, zspec={zb:.3f}',fontsize=10)
        #plt.text(x=0.6,y=0.8,s=f'{self.fname}\n z={zb:.3f}',
         #                        bbox = dict(facecolor = 'white', alpha = 0.5),fontsize=10,
          #       transform=ax.transAxes)
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
    
    
    def get_lines(self,z,halpha_prism='fixed',get_broad=False):

        # broad lines

        allowed_bl = ['Ha','Hb', 'Hg', 'Hd','H8','H9', 'H10', 'H11', 'H12','HeI-1083','HeI-3889','HeI-5877','HeI-6680','HeI-7065', 'HeI-8446','HeII-1640','HeII-4687']
        
        # templates = {}
        
        
        if self.grating == 'prism':
            hlines = ['Hb', 'Hg', 'Hd']
            
            if z > 4:
                oiii = ['OIII-4959','OIII-5007']
                hene = ['HeII-4687', 'NeIII-3867','HeI-3889']
                o4363 = ['OIII-4363']
                
            else:
                oiii = ['OIII']
                hene = ['HeI-3889']
                o4363 = []
                
            sii = ['SII']
            #sii = ['SII-6717', 'SII-6731']
            
            if halpha_prism=='fixed':
                halpha_prism = ['Ha+NII']
            else:
                halpha_prism = ['Ha','NII']
            
            hlines += halpha_prism + ['NeIII-3968']
            fuv = ['OIII-1663']
            oii_7320 = ['OII-7325']
            extra = []
            
        else:
            hlines = ['Hb', 'Hg', 'Hd','H8','H9', 'H10', 'H11', 'H12']
            
            hene = ['HeII-4687', 'NeIII-3867']
            o4363 = ['OIII-4363']
            oiii = ['OIII-4959','OIII-5007']
            sii = ['SII-6717', 'SII-6731']
            hlines += ['Ha', 'NII-6549', 'NII-6584']
            hlines += ['H7', 'NeIII-3968']
            fuv = ['OIII-1663', 'HeII-1640', 'CIV-1549']
            oii_7320 = ['OII-7323', 'OII-7332']
            
            extra = ['HeI-6680', 'SIII-6314']
            
        line_list =  [*hlines, *oiii, *o4363, 'OII',
                  *hene, 
                  *sii,
                  *oii_7320,
                  'ArIII-7138', 'ArIII-7753', 'SIII-9068', 'SIII-9531',
                  'OI-6302', 'PaD', 'PaG', 'PaB', 'PaA', 'HeI-1083',
                  'BrA','BrB','BrG','BrD','PfB','PfG','PfD','PfE',
                  'Pa8','Pa9','Pa10',
                  'HeI-5877', 
                  *fuv,
                  'CIII-1908', 'NIII-1750', 'Lya',
                  'MgII', 'NeV-3346', 'NeVI-3426',
                  'HeI-7065', 'HeI-8446',
                  *extra]
        
        if get_broad==True:
            return list(set(line_list).intersection(allowed_bl))
        else:
            return line_list

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
        import corner
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
