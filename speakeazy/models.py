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
from priors import Priors


rc('axes', linewidth=1.5)
rc('xtick',direction='in')#, minor_visible=True, major_width=1.2, minor_width=1.0)
rc('ytick',direction='in')#, minor_visible=True, major_width=1.2, minor_width=1.0)
rc('font',size=14)
plt.rcParams["font.family"] = "serif"

class Models(object):
    """Models 

    _extended_summary_

    Arguments:
        object -- _description_
        
    Attributes:
        xline -- 
        yline -- 
        lw -- 
        lr --
        resample_func -- 
        bsplines --  
        
    
    
    Functions:
        initialize_emission_line -- make delta functions to make into lines
        emission_line -- generate gaussian emission line that is resampled to observed spectral resolutioon
        bspline_array -- generate spline templates for continuum 
        initialize_bsplines -- make the continnuum templates and store 
        generate_templates -- make gaussian emission line templates and splines and return array of templates with specific vel_width,scale,scale_disp and z 
        get_lines -- get line centers and names of lines 
    """
    
    def __init__(self):
        
        reload(msaexp.resample_numba); reload(msaexp.spectrum)
        reload(msaexp.resample_numba); reload(msaexp.spectrum)

        
        try:
            from msaexp.resample_numba import \
                resample_template_numba as resample_func
        except ImportError:
            from .resample import resample_template as resample_func
            
        self.resample_func = resample_func
        # make emission line templates as delta functions for now 
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
        res = self.resample_func(Data.spec_wobs,
                                 Data.spec_R_fwhm*scale_disp, 
                                 self.xline*line_um,
                                 self.yline,
                                 velocity_sigma=velocity_sigma,
                                 nsig=nsig)
        return res*line_flux/line_um

    def bspline_array(self, nspline=13, log=False):
        """
        Generate bspline for continuum models 
        """
        bspl = utils.bspline_templates(wave=Data.spec_wobs*1.e4,
                                       degree=3,
                                       df=nspline,
                                       log=log,
                                       get_matrix=True
                                       )
        return bspl.T

    def initialize_bsplines(self,input_params):
        """ Make the bspline arrays """
        if Priors.params['nspline'] is not None:
            self.bsplines = self.bspline_array(nspline=Priors.params['nspline'],log=False) #initialise
        else:
            # set default to 21 
            self.bsplines = self.bspline_array(nspline=21,log=False) #initialise
            

    def generate_templates(self,z,scale_disp,vel_width,vel_width_b,theta=None,init=False,broadlines=False):
        """
        Generates gaussian emission line templates.

        Parameters 
        ----------
        
        z: float
            redshift

        scale_disp: float
            scale dispersion

        vel_width: float
            line velocity width km/s

        vel_width_b: float
            broad line velocity width km/s

        init: bool
            True if first run, false if used during likelihood fitting and mcmc 

        Returns
        ---------- 
        flux_arr: array [NTEMP, NWAVE] [erg/s/cm2??]
            Array of templates where NTEMP = nsplines + nlines, NWAVE = len(spectrum wavelength array)

        line_names_: list of line names, only returned if init=True 
        """

        broadlines = Priors.params['broadlines']
        lines = []
        # broad
        broad_lines = []
        tline = []
       # tline1 = []
   
        
        if init:
            line_names = self.get_lines(z,Priors.params['halpha_prism'],get_broad=False)
            line_names_ = []
            # broad
            
            if broadlines:
                broad_line_names = self.get_lines(z,Priors.params['halpha_prism'],get_broad=True)
                broad_line_names_ = []
            
            
        else:

            line_names1 = [l for l in theta.keys() if l.startswith("line ")]
       
            line_names = [re.sub(r'line ', '', l,1) for l in line_names1]
        
            # broad
            if broadlines:
                broad_line_names1 = [l for l in self.theta.keys() if l.startswith("broad line ")]
                broad_line_names = [re.sub(r'broad line ', '', l,1) for l in broad_line_names1]
   

     
            
        for k in line_names:
            if (k not in self.lw):
                continue
            else:
                line_k = np.zeros(len(Data.spec_wobs))
                for lwk, lrk in zip(self.lw[k], self.lr[k]):
                    line_k += self.emission_line(lwk*(1.+z)/1.e4,
                                                 line_flux=lrk/np.sum(self.lr[k]),
                                                 scale_disp=scale_disp,
                                                 velocity_sigma=vel_width,
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
                    line_kb = np.zeros(len(Data.spec_wobs))
                    for lwk, lrk in zip(self.lw[kb], self.lr[kb]):
                        line_kb += self.emission_line(lwk*(1.+z)/1.e4,
                                                     line_flux=lrk/np.sum(self.lr[kb]),
                                                     scale_disp=scale_disp,
                                                     velocity_sigma=vel_width_b,
                                                     nsig=4)
                    # broad
                    broad_lines.append(line_kb)
                    if init:
                        broad_line_names_.append(kb)
                    tline.append(1)

        Priors.params['nlines'] = (np.array(lines).shape)[0]
        # broad
        if broadlines:
            Priors.params['nblines'] = (np.array(broad_lines).shape)[0] 
        else:

            Priors.params['nblines'] = 0
            
        NTEMP = Priors.params['nlines']+Priors.params['nblines']+Priors.params['nspline']
        flux_arr = np.zeros((NTEMP, Data.NWAVE))
        
        flux_arr[0:Priors.params['nlines'],:] = np.array(lines)/1.e4

        if broadlines:

            flux_arr[Priors.params['nlines']:Priors.params['nlines']+Priors.params['nblines'],:] = np.array(broad_lines)/1.e4
        
        if init:
            flux_arr[Priors.params['nlines']+Priors.params['nblines']:,:] = self.bsplines
            for i in range(self.bsplines.shape[0]):
                tline.append(2)
        else:
            flux_arr[Priors.params['nlines']+Priors.params['nblines']:,:] = self.bsplines[self.spl_mask]
            for i in range(self.bsplines[self.spl_mask].shape[0]):
                    tline.append(2)

        if not broadlines:
            broad_line_names_ = []

        
        if init:
            return flux_arr, line_names_, broad_line_names_, np.array(tline)
        else:
            return flux_arr,np.array(tline)
        
    def get_lines(self,z,halpha_prism='fixed',get_broad=False):

        # broad lines

        allowed_bl = ['Ha','Hb', 'Hg', 'Hd','H8','H9', 'H10', 'H11', 'H12','HeI-1083','HeI-3889','HeI-5877','HeI-6680','HeI-7065', 'HeI-8446','HeII-1640','HeII-4687']
        
        # templates = {}
        
        
        if Data.grating == 'prism':
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
        