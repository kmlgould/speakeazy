# import modules -------------------------------------------------------------------------------------------
import os
import re
import hickle as hkl
import msaexp
import numpy as np
import sys
import warnings

import astropy.units as u
import eazy
import numba
from astropy.io import fits
from grizli import utils as utils
from pylab import *
from collections import OrderedDict
from pylab import *
from astropy.table import Table
from msaexp import pipeline, spectrum, resample_numba
from importlib import reload

utils.set_warnings()
print(f"msaexp version = {msaexp.__version__}")
print(f"numpy version = {np.__version__}")

#--------------------------------------------------------------------------------------------------------------

class Data(object):
    """Data loads and stores 1D spectral data as well as broad-band photometry

    _extended_summary_

    Arguments:
        object -- _description_
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