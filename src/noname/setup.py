from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Package for fitting JWST prism spectra'
LONG_DESCRIPTION = 'Bleurghhhhhhh'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="noname", 
        version=VERSION,
        author="Katriona Gould",
        author_email="<katriona.gould@nbi.ku.dk>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Astronomy",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)