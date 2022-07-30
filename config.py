# -*- coding: utf-8 -*-
"""
Dictionaries containing the relation between world coordinates.
"""


# setting the world types
world_types = {
    'RA':'fk5',
    'DEC':'fk5',
    'GLON':'galactic',
    'GLAT':'galactic'
    }

#The labels associated with each coordinate type
coord_types = {
    'fk5': ['RA', 'DEC'],
    'galactic':['GLON', 'GLAT'],
    'pixel':['x', 'y']
    }

#Will only convert between galactic and equatorial. This dict encodes
#if our world coords are `fk5`, the world conversion is to `galactic`
coord_conversion = {
    'fk5':'galactic',
    'galactic':'fk5',
    'pixel':'pixel'
    }

if __name__ == "__main__":
    pass
