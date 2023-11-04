import numpy as np 
import os

from .. import data,priors,sampling,fitting


class TestSampler():
    def setup(self):
         spec_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
         _spec_file = os.path.join(spec_dir, 
                                          'macs0417.1208_340.v0.spec.fits')
         spectrum = data.Data(spectrum_file=_spec_file,photometry_file=None,run_ID=1,phot_id=None)
         prs = priors.Priors(spectrum)
         
         fit_object = fitting.Fitter(spectrum,prs)
         fit_object.fit_redshift_chisq()
         fit_object.plot_spectrum()