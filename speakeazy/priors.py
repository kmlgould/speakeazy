import os

import numpy as np

from grizli import utils

from .data import Data

class Priors(object):
    """Priors _summary_

    _extended_summary_

    Arguments:
        object -- _description_
        
    Attributes:
    
    
    Functions:
    
    to sample for joint prior distributions, perhaps use either inverse transformation method, or accept reject method. 
    """
    
    def __init__(self,data,fix_ns=True,nspline=13,epoly=3,ppoly=3,vel_width=100.,vel_width_broad=300.,scale_disp=1.3,z=None,z0=[0.,6.],halpha_prism='free',scale_p=False,broadlines=False,prior_instructions={'redshift_prior':'norm',
                                            'z_mu' : 1., 
                                            'z_sigma' : 0.01,
                                            'velocity_prior':'norm', 
                                            'v_mu':300.,
                                            'v_sigma':100.,
                                            'broad_velocity_prior':'norm', 
                                            'vb_mu':2000., 
                                            'vb_sigma':500.,
                                            'error_scale_prior':'norm',
                                            'escale_mu':2.,
                                            'escale_sigma':0.1,
                                            'scale_disp_prior':'norm',
                                            'scale_disp_mu':1.3,
                                            'scale_disp_sigma':0.1},input_fixed_lines=[]):
        lw, lr = utils.get_line_wavelengths()
        self.data = data 
        self.lw = lw
        self.lr = lr
        self.set_params(fix_ns,nspline,epoly,ppoly,vel_width,vel_width_broad,scale_disp,z,z0,halpha_prism,scale_p,broadlines)    
        self.set_priors(prior_instructions)
        self.set_hlines_prior()
        self.set_fixed_lines(input_fixed_lines)
        
    def set_fixed_lines(self,input_fixed_lines=[]):
        
        
        if self.data.grating == 'prism':
            default_lines = ['Ha','NII','OIII','Hb']
        else:
            default_lines = ['Ha','NII-6549', 'NII-6584','OIII-4959','OIII-5007','Hb']
           
        fixed_lines = np.concatenate(default_lines,input_fixed_lines)
        
        self.fixed_lines = fixed_lines
        
    def set_params(self,fix_ns=True,nspline=13,epoly=3,ppoly=3,vel_width=100.,vel_width_broad=300.,scale_disp=1.3,z=None,z0=[1.4,1.5],halpha_prism='free',scale_p=False,broadlines=False):
        
        self.params = {}
        self.params['z_in'] = z
        self.params['z_range'] = z0
        self.params['scale_disp']=scale_disp
        self.params['vel_width'] = vel_width
        self.params['vel_width_broad'] = vel_width_broad
        self.params['fix_ns']=fix_ns
        self.params['nspline']=nspline
        self.params['epoly']=epoly
        self.params['scale_p']=scale_p
        if self.params['scale_p']==True:
            self.params['ppoly'] = ppoly
        else:
            self.params['ppoly'] = 0
            
        self.params['halpha_prism']=halpha_prism
        self.params['broadlines']=broadlines
        
    @staticmethod
    def create_prior(prior_type='norm',mu=0.,sigma=1.):
        """create_prior create either a gaussian or flat prior


        Keyword Arguments:
            prior_type -- str (default: {'norm'})
            mu -- mean if gaussian, lower edge if uniform
            sigma -- std if gaussian, upper edge if uniform
        """
        from scipy.stats import expon, gamma, norm, uniform
        
        if prior_type=='norm':
            prior_dist = norm(loc=mu,scale=sigma)
        if prior_type=='flat':
            prior_dist = uniform(loc=mu,scale=sigma)
            
        return prior_dist     
        
        
    def set_priors(self,prior_instructions={'redshift_prior':'norm',
                                            'z_mu' : 1., 
                                            'z_sigma' : 0.01,
                                            'velocity_prior':'norm', 
                                            'v_mu':300.,
                                            'v_sigma':100.,
                                            'broad_velocity_prior':'norm', 
                                            'vb_mu':2000., 
                                            'vb_sigma':500.,
                                            'error_scale_prior':'norm',
                                            'escale_mu':2.,
                                            'escale_sigma':0.1,
                                            'scale_disp_prior':'norm',
                                            'scale_disp_mu':1.3,
                                            'scale_disp_sigma':0.1}):
        
        """set_priors set prior distributions based on dictionary given by user

        _extended_summary_
        """

        
        self.z_rv = self.create_prior(prior_instructions['redshift_prior'],prior_instructions['z_mu'],prior_instructions['z_sigma'])
        self.vw_rv = self.create_prior(prior_instructions['velocity_prior'],prior_instructions['v_mu'],prior_instructions['v_sigma'])
        self.vwb_rv = self.create_prior(prior_instructions['broad_velocity_prior'],prior_instructions['vb_mu'],prior_instructions['vb_sigma'])
        self.escale_rv=self.create_prior(prior_instructions['error_scale_prior'],prior_instructions['escale_mu'],prior_instructions['escale_sigma'])
        self.sc_rv = self.create_prior(prior_instructions['scale_disp_prior'],prior_instructions['scale_disp_mu'],prior_instructions['scale_disp_sigma'])     
        
        
    def plot_priors(self,nexp=1000):
        
        import matplotlib.pyplot as plt
        
        fig,axes = plt.subplots(ncols=5,nrows=1,figsize=(12,4))
        
        axes[0].hist(self.z_rv.rvs(nexp))
        axes[0].set_xlabel('z')
        
        axes[1].hist(self.vw_rv.rvs(nexp))
        axes[1].set_xlabel('v [km/s]')
        
        axes[2].hist(self.vwb_rv.rvs(nexp))
        axes[2].set_xlabel('v broad [km/s]')
        
        axes[3].hist(self.escale_rv.rvs(nexp))
        axes[3].set_xlabel('error scaling')
        
        axes[4].hist(self.sc_rv.rvs(nexp))
        axes[4].set_xlabel('scale dispersion')
        
        plt.show()

    
    def set_hlines_prior(self):
        from scipy.stats import norm

        # Balmer line ratios for Ha,Hb,Hg,Hd, prior based on case B, ratios from Groves et al. 2011
        # https://arxiv.org/pdf/1109.2597.pdf

        hahb_lr = self.lr['Balmer 10kK'][0] 
        hahg_lr = self.lr['Balmer 10kK'][0]*(1./self.lr['Balmer 10kK'][2])
        hahd_lr = self.lr['Balmer 10kK'][0]*(1./self.lr['Balmer 10kK'][3])
        #print("2.86, 0.47, 0.26")
        #print(hahb_lr,hghb_lr,hdhb_lr)
        
        self.hahb_rv = norm(loc=hahb_lr,scale=1.)
        self.hahg_rv = norm(loc=hahg_lr,scale=1.)
        self.hahd_rv = norm(loc=hahd_lr,scale=1.)
    
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
    
    #def measure_line_prior():
    #    """line_prior measures FWHM of lines and draws vel width and scale disp from joint prior distribution 
    #    takes as input: raw data? ehhhh 
    #    _extended_summary_
    #    """