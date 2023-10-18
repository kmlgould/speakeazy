# import modules -------------------------------------------------------------------------------------------
import os
import re
import sys
import warnings
from collections import OrderedDict

import hickle as hkl
import numpy as np
import astropy.units as u
import eazy

from astropy.io import fits
from grizli import utils

from pylab import *
from astropy.table import Table
import msaexp

from importlib import reload

#--------------------------------------------------------------------------------------------------------------

class Data(object):
    """Data loads and stores 1D spectral data as well as broad-band photometry

    _extended_summary_

    Arguments:
        object -- _description_
        
    Attributes:

        
    Functions:
        initialize_spec -- Read in 1D spectrum file and create spectrum attributes 
    """
    
    def __init__(self,spectrum_file,photometry_file,run_ID,phot_id):

        try:
            from msaexp.resample_numba import resample_template_numba as resample_func
        except ImportError:
            from msaexp.resample import resample_template as resample_func
            
        self.resample_func = resample_func
        self.spectrum_file = spectrum_file
        self.photometry_file = photometry_file
        self.run_ID = run_ID
        self.phot_id = phot_id
        
        
        # create new folder for this data and session 
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
        
        self.initialize_spec()

        if self.photometry_file:
            self.initialize_phot()
        else:
            print("No photometry found")

        self.initialize_emission_line()
        
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
        res = self.resample_func(self.spec_wobs,
                                 self.spec_R_fwhm*scale_disp, 
                                 self.xline*line_um,
                                 self.yline,
                                 velocity_sigma=velocity_sigma,
                                 nsig=nsig)
        return res*line_flux/line_um
    
    def bspline_array(self, nspline=13, log=False):
        """
        Generate bspline for continuum models 
        """
        bspl = utils.bspline_templates(wave=self.spec_wobs*1.e4,
                                       degree=3,
                                       df=nspline,
                                       log=log,
                                       get_matrix=True
                                       )
        return bspl.T
        
       
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

                #if self.xwave is not None:

                #    vvalid &= (spec['wave']>self.xwave[0]) & (spec['wave']<self.xwave[1])


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

                
                _disp_path = os.path.join(os.path.dirname(__file__), 
                                          'data/msa_nirspec_disp_curves')
                _disp_file = os.path.join(_disp_path, 
                                          f'jwst_nirspec_{grating}_disp.fits')
                
                disp = utils.read_catalog(_disp_file)

                #spec['R'] = np.interp(spec['wave'], disp['WAVELENGTH'], disp['R'],
                #                      left=disp['R'][0], right=disp['R'][-1])
                
                flam_unit = 1.e-19*u.erg/u.second/u.cm**2/u.Angstrom # change to attribute 
                self.equiv = u.spectral_density(spec['wave'].data*um)
                self.to_flam = (1.*spec['flux'].unit).to(flam_unit, equivalencies=self.equiv).value #property 
                self.flamunit = flam_unit.unit
                self.spec_R_fwhm = (np.interp(spec['wave'], disp['WAVELENGTH'], disp['R'],
                                      left=disp['R'][0], right=disp['R'][-1])).astype(np.float32)
                self.valid = spec['valid']
                self.spec_wobs = spec['wave'].value.astype(np.float32)
                self.spec_fnu = spec['flux'].value.astype(np.float32)
                self.spec_efnu = spec['err'].value.astype(np.float32)
                self.grating = grating
                self.filter = _filter


            # otherwise assume its from the ETC

            else:
                spec = im[1].data
                um = u.micron
                wave = spec['WAVELENGTH']
                fluxin = spec['extracted_flux_plus_bg_resel']
                flux = np.array([self.flam_to_ujy(fl*1e-19,ww*1e4) for (fl,ww) in zip(fluxin,wave)])

                _data_path = os.path.join(os.path.dirname(__file__), 
                                          'data/msa_nirspec_disp_curves')
                
                grating = "prism"
                _filter = "clear"
                _disp_file = os.path.join(_data_path, 
                                          f'jwst_nirspec_{grating}_disp.fits')
                disp = utils.read_catalog(_disp_file)
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
