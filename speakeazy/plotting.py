import os
import re
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from astropy.table import Table
import astropy.units as u
from astropy.io import fits

from .fitting import Fitting 
from .priors import Priors
from .data import Data
from .plotting import Plotting

class Plotting(object):
    """Plotting 

    _extended_summary_

    Arguments:
        object -- _description_
    """
    
    def __init__(self) -> None:
        pass 
    
    def plot_spectrum(self,save=True,fname=None,flat_samples=None,line_snr=5.,show_lines=False,ylims=None,xlims=None):
        mask = Data.valid
        
        flam = Data.spec_fnu*Data.to_flam
        eflam = Data.spec_efnu*Data.to_flam
        
        flam[~mask] = np.nan
        eflam[~mask] = np.nan
        
        wav = Data.spec_wobs
        
        xmin = np.nanmin(wav[mask])
        xmax = np.nanmax(wav[mask])

        plt.figure(figsize=(12,4))
        #plt.fill_between(wav,(flam+eflam),((flam-eflam)),\
                                # alpha=0.4,color='cornflowerblue',zorder=-99)

        scale=1.
        
        if (Priors.params['scale_p']==True):
            if (Fitting.theta['pscale_0']!=1.) : 
                pscale_coeffs = np.array([Fitting.theta[f'pscale_{i}'] for i in range(Priors.params['ppoly'])])
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

        if hasattr(Fitting, 'model_spec'):
            #if (self.params['scale_p']==True):
            #    if (self.theta['pscale_0']!=1.):
                    # scaling has already been applied 
          #          plt.step(wav[mask],(self.model_spec)[mask],color='blue',label='Model')
        #else:
            plt.step(wav[mask],(Fitting.model_spec*scale)[mask],color='black',label='Model')

            
            plt.step(wav[mask],(Fitting.model_line*scale)[mask],color='blue',label='Lines')
            #plt.step(wav[mask],(self.model_bline*scale)[mask],color='red',label='Broad lines')
            #plt.step(wav[mask],self.model_cont[mask],color='olive',label='Continuum')

            if hasattr(scale, "__len__"):
                plt.plot(Data.spec_wobs[mask], (((Fitting.Acont.T).T))*scale[mask,None],
                        color='olive', alpha=0.3)
            else:
                plt.plot(Data.spec_wobs[mask], (((Fitting.Acont.T).T))*scale,
                        color='olive', alpha=0.3)
            # plot emission lines 

            if show_lines:

                
                
                for line in Fitting.line_table:
                    l_snr = abs(Fitting.line_table[line][0])/abs(Fitting.line_table[line][1])
                    if l_snr>line_snr:
                        #lname = line.strip(' line') # get name of line
                        lname = re.sub(r'line ', '', line)
                        if len(Priors.lw[lname])>0:
                            wavl = np.average(Priors.lw[lname])
                        else:
                            wavl = Priors.lw[lname]
                        line_center = (wavl*(1.+Fitting.theta['z']))/1e4
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
    

