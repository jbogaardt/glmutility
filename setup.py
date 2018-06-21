descr = "GLM (generalized linear models) Utility"

#from distutils.core import setup
from setuptools import setup

setup(
    name='GLMUtility',
    version='0.1.1',
    maintainer='John Bogaardt',
    maintainer_email='jbogaardt@wcf.com',
    packages=['GLMUtility'],
    scripts=[],
    url='https://github.com/jbogaardt/GLMUtility',
    download_url='',
    license= 'https://github.com/jbogaardt/GLMUtility/blob/master/LICENSE',

    description= descr,

    install_requires=[
        "bokeh>=0.12.15",
        "numpy>=1.14.3",
        "pandas>=0.22.0",
        "ipywidgets>=7.2.1"
    ],
)
