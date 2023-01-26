# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:06:22 2023

@author: Bennett
"""

import numpy as np # Import package that contains efficient data storage structures
import astropy.units as u # Import package that allows units to be tracked explicity
from ReadFile import Read # Import function that reads files

def ParticleInfo(filename, particle_type, particle_number):
    '''
    Gets the distance from the center of mass of the Milky Way, velocity magnitude, and mass
    of the particle_numberth particle (1-indexed) of type particle_type
    @param filename (str), the name of the file where the data is stored
    @param particle_type (float), the type of particle data is desired for (1.0=Dark Matter, 2.0=Disk Stars, 3.0=Buldge Stars)
    @param particle_number (int), the number particle of particle type to get data for
    @returns dist (Quantity), the distance of the particle from the center of the Milky Way
    @returns vel (Quantity), the velocity of the particle
    @returns mass (Quantity), the mass of the particle
    '''
    particle_number -= 1 # Convert from 1-indexed numbers used by people to 0-index used by Python
    time, num_particles, data = Read(filename) # Read in the file
    indicies = np.where(data['type'] == particle_type)[0] # Get a list of the indicies of particles w/ a given type
    index = indicies[particle_number] # Get the index of the desired particle
    dist = np.sqrt(float(data['x'][index])**2+float(data['y'][index])**2+float(data['z'][index])**2)*u.kpc # Get particle distance from MW COM (given in kpc)
    vel = np.sqrt(float(data['vx'][index])**2+float(data['vy'][index])**2+float(data['vz'][index])**2)*u.km/u.s # Get magnitude of the particle velocity (given in km/s)
    mass = float(data['m'][index])*1e10*u.solMass # Get mass of the particle (given in 1e10 solar masses)
    dist, vel = np.around(dist, 3), np.around(vel, 3) # Round distance and velocity values
    return dist, vel, mass # Return the distnace, velocity, and mass of the particle