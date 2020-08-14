from setuptools import setup

setup(
    name='gym_ev',
    version='0.0.1',
    description='Gym walk environment - useful to replicate Random Walk experiments',
    url='https://github.com/ercosv/gym-ev',
    author='EV',
    packages=['gym_ev', 'gym_ev.env'],
    install_requires=['gym'],
)
