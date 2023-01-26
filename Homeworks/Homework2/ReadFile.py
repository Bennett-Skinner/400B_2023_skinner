# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:46:22 2023

@author: Bennett
"""

import numpy as np # Import package that contains efficient data storage structures
import astropy.units as u # Import package that allows units to be tracked explicity

def Read(filename):
    '''
    Reads in the data from a file , assuming it has a header that gives its time and # of particles
    @param filename (str), the name of the file
    @returns time (Quantity), the time of the simulation stored in this file
    @returns num_particles (int), the number of particles in the simulation stored in this file
    @returns data (ndarray), the data stored in this file (meaning of data in column labels)
    '''
    file = open(filename,'r') # Open file to be read
    line1 = file.readline() # Read first line
    label, value1 = line1.split() # Obtain the label and value of what is stored in the first line
    time = float(value1)*u.Myr # Store the time of this file
    line2 = file.readline() # Read second line
    label2, value2 = line2.split() # Obtain the label and value of what is stored in the second line
    num_particles = int(value2) # Store the total number of particles in this file
    file.close() # Close file
    data = np.genfromtxt(filename,dtype=None,names=True,skip_header=3) # Store data in the file
    return time, num_particles, data # Return the time of, number of particles in, and data in file