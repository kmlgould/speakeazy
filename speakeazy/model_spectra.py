import os
import re
import sys
import time
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table
import astropy.units as u
from astropy.io import fits

from grizli import utils

utils.set_warnings()

# Make what you need from pylab implicit!
from pylab import *

from .priors import Priors

rc('axes', linewidth=1.5)
rc('xtick',direction='in')#, minor_visible=True, major_width=1.2, minor_width=1.0)
rc('ytick',direction='in')#, minor_visible=True, major_width=1.2, minor_width=1.0)
rc('font',size=14)
plt.rcParams["font.family"] = "serif"

class Model(object):
    """Model

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
        bspline_array -- generate spline templates for continuum 
        initialize_bsplines -- make the continnuum templates and store 
        generate_templates -- make gaussian emission line templates and splines and return array of templates with specific vel_width,scale,scale_disp and z 
        get_lines -- get line centers and names of lines 
    """
    
    def __init__(self,data,priors):
        
        self.nspline = priors.params['nspline']
        self.fit_broadlines = priors.params['broadlines']
        self.halpha_prism = priors.params['halpha_prism']
        
        self.spec_wobs = data.spec_wobs
        self.NWAVE = data.NWAVE
        self.grating = data.grating 
        
        lw, lr = utils.get_line_wavelengths()

        self.lw = lw
        self.lr = lr
        
        self.initialize_bsplines()

        # make emission line templates as delta functions for now 

    def initialize_bsplines(self,default_nspline=21):
        """initialize_bsplines _summary_

        _extended_summary_

        Keyword Arguments:
            default_nspline -- _description_ (default: {21})
        """
        if self.nspline is not None:
            self.bsplines = self.bspline_array(nspline=self.nspline,log=False) #initialise
        else:
            # set default to 21 
            self.bsplines = self.bspline_array(nspline=default_nspline,log=False) #initialise
            
    def bspline_array(self, nspline=21, log=False):
        """bspline_array _summary_

        _extended_summary_

        Keyword Arguments:
            nspline -- _description_ (default: {21})
            log -- _description_ (default: {False})

        Returns:
            _description_
        """
        bspl = utils.bspline_templates(wave=self.spec_wobs*1.e4,
                                       degree=3,
                                       df=nspline,
                                       log=log,
                                       get_matrix=True
                                       )
        return bspl.T
            

    def generate_templates(self,data,z=None,scale_disp=1.3,vel_width=100.,vel_width_broad=1000.,theta=None,chisq=False,broadlines=False):
        """
        Generates a spectrum template comprised of gaussian emission line templates and continuum based on spline functions 

        Parameters 
        ----------
        
        z: float
            redshift

        scale_disp: float
            scale dispersion

        vel_width: float
            line velocity width km/s

        vel_width_broad: float
            broad line velocity width km/s

        chisq: bool
            True if simple chi-square fit, false if used during likelihood fitting and mcmc 

        Returns
        ---------- 
        flux_arr: array [NTEMP, NWAVE] [erg/s/cm2??]
            Array of templates where NTEMP = nsplines + nlines, NWAVE = len(spectrum wavelength array)

        line_names_: list of line names, only returned if init=True - should depracate this. 
        """

        ############################################################################################################
        ##################### EMISSION LINE TEMPLATES ##############################################################
        
        
        broadlines = self.fit_broadlines
        lines = []
        # broad
        broad_lines = []
        tline = []
       # tline1 = []
   
        
        if chisq:
            line_names = self.get_lines(z,self.halpha_prism,get_broad=False)
            line_names_ = []
            # broad
            
            if broadlines:
                broad_line_names = self.get_lines(z,self.halpha_prism,get_broad=True)
                broad_line_names_ = []
            
            
        else:

            line_names_pre = [l for l in theta.keys() if l.startswith("line ")]
       
            line_names = [re.sub(r'line ', '', l,1) for l in line_names_pre]
        
            # broad
            if broadlines:
                broad_line_names_pre = [l for l in self.theta.keys() if l.startswith("broad line ")]
                broad_line_names = [re.sub(r'broad line ', '', l,1) for l in broad_line_names_pre]
   

     
            
        for k in line_names:
            if (k not in self.lw):
                continue
            else:
                line_k = np.zeros(len(self.spec_wobs))
                for lwk, lrk in zip(self.lw[k], self.lr[k]):
                    line_k += data.emission_line(lwk*(1.+z)/1.e4,
                                                 line_flux=lrk/np.sum(self.lr[k]),
                                                 scale_disp=scale_disp,
                                                 velocity_sigma=vel_width,
                                                 nsig=4)
                lines.append(line_k)
                
                if chisq:
                    line_names_.append(k)
                    #tline.append(True)
                tline.append(0)

        if broadlines:
             
            for kb in broad_line_names:
                if kb not in self.lw:
                    continue
                else:
                    line_kb = np.zeros(len(self.spec_wobs))
                    for lwk, lrk in zip(self.lw[kb], self.lr[kb]):
                        line_kb += data.emission_line(lwk*(1.+z)/1.e4,
                                                     line_flux=lrk/np.sum(self.lr[kb]),
                                                     scale_disp=scale_disp,
                                                     velocity_sigma=vel_width_broad,
                                                     nsig=4)
                    # broad
                    broad_lines.append(line_kb)
                    if chisq:
                        broad_line_names_.append(kb)
                    tline.append(1)

        self.priors.params['nlines'] = (np.array(lines).shape)[0]
        # broad
        if broadlines:
            self.priors.params['nblines'] = (np.array(broad_lines).shape)[0] 
        else:

            self.priors.params['nblines'] = 0
            
        NTEMP = self.priors.params['nlines']+self.priors.params['nblines']+self.nspline
        flux_arr = np.zeros((NTEMP, self.data.NWAVE))
        
        flux_arr[0:self.priors.params['nlines'],:] = np.array(lines)/1.e4

        if broadlines:

            flux_arr[self.priors.params['nlines']:self.priors.params['nlines']+self.priors.params['nblines'],:] = np.array(broad_lines)/1.e4
        
        if chisq:
            flux_arr[self.priors.params['nlines']+self.priors.params['nblines']:,:] = self.bsplines
            for i in range(self.bsplines.shape[0]):
                tline.append(2)
        else:
            flux_arr[self.priors.params['nlines']+self.priors.params['nblines']:,:] = self.bsplines[self.spl_mask]
            for i in range(self.bsplines[self.spl_mask].shape[0]):
                    tline.append(2)

        if not broadlines:
            broad_line_names_ = []

        
        if chisq:
            return flux_arr, line_names_, broad_line_names_, np.array(tline)
        else:
            return flux_arr,np.array(tline)
        
    def get_lines(self,z,halpha_prism='fixed',get_broad=False): # remove halpha_prism from input? 
        """get_lines _summary_

        _extended_summary_

        Arguments:
            z -- _description_

        Keyword Arguments:
            halpha_prism -- _description_ (default: {'fixed'})
            get_broad -- _description_ (default: {False})

        Returns:
            _description_
        """

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
            
            if self.halpha_prism=='fixed':
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
        
