import setuptools

# with open("README.md", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name='covid_abs',
    packages=['covid_abs','covid_abs.network'],
    version='0.0.2',
    description='COVID-19 Agent Based Simulation',
    long_description='Agent Based Simulation of COVID-19 Health and Economical Effects',
    long_description_content_type="text/markdown",
    author='Petronio Candido L. e Silva',
    author_email='petronio.candido@gmail.com',
    url='https://github.com/petroniocandido/COVID19_AgentBasedSimulation',
    download_url='https://github.com/petroniocandido/COVID19_AgentBasedSimulation',
    keywords=['covid19', 'simulation', 'agent based simulation'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'Development Status :: 5 - Production/Stable'

    ]
)