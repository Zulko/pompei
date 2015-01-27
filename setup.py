try:
    from setuptools import setup
except ImportError:
    try:
        import ez_setup
        ez_setup.use_setuptools()
    except ImportError:
        raise ImportError("Vapory could not be installed, probably because"
            " neither setuptools nor ez_setup are installed on this computer."
            "\nInstall ez_setup ([sudo] pip install ez_setup) and try again.")

from setuptools import setup, find_packages

exec(open('pompei/version.py').read()) # loads __version__

setup(name='pompei',
      version=__version__,
      author='Zulko',
    description='Create Mosaics from movie frames',
    long_description=open('README.rst').read(),
    license='see LICENSE.txt',
    keywords="mosaic movie frame moviepy",
    packages= find_packages(exclude='docs'),
    install_requires= ['moviepy'])
