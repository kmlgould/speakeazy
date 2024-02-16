import numpy as np 
import os

from .. import data,priors,sampling,fitting


class TestSampler():
    def setup(self):
         spec_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
         _spec_file = os.path.join(spec_dir, 
                                          'macs0417.1208_340.v0.spec.fits')
         self.spectrum = data.Data(spectrum_file=_spec_file,photometry_file=None,run_ID=1,phot_id=None)
         self.priors = priors.Priors(self.spectrum)
         
         
    def test_filenames(self):
         " make sure a file is created with the run ID and all figures and outputs are saved to that file."
         self.spectrum.fname = 'macs0417.1208_340.v0'
         print(self.spectrum.fname)
         print(self.spectrum.ID)

    def test_fitting(self):
         fit_object = fitting.Fitter(self.spectrum,self.priors)
         fit_object.fit_redshift_chisq()
         fit_object.data.fname = 'macs0417.1208_340.v0'
         fit_object.plot_spectrum(save=True)

         test_path = f'{fit_object.data.run_ID}_initial_fit.png'
        
         check = os.path.exists(test_path)
        
         assert check==True 