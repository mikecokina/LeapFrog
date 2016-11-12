#!/usr/bin/python
import time
import numpy as np
import scipy as sp
from scipy import constants
import sys

import Leapfrog as leapfrog
import Vis as vis

import random
def rand(mn, mx, c, seed):
	random.seed(seed)
	return [mn + random.random() * (mx - mn) for i in xrange(c)]


# all units in convention of Gauss grav. constant k = 0.01720
# [M] = M_sun, [r] = AU, [v] = AU/d
"""
lf = leapfrog.Leapfrog(m = [1.9891e30, 5.97219e24],
			  r = [[0., 0., 0.], [sp.constants.au, 0., 0.]],
			  v = [[0., 0., 0.], [0., 40.*3600, 0.]])
"""

"""
lf = leapfrog.Leapfrog(m = [0.10000000000000000e+01, 0.28583678719451197e-03],
			  r = [[0., 0., 0.], [0.69905175661092100e+00, -0.95522604421347300e+01, -0.43673105309042500e-01]],
			  v = [[0., 0., 0.], [0.55414811016520800e-02, 0.29850559025851500e-03, -0.22127242395468300e-03]])
"""

m, r, v = [3.], [[0.,0.,0.]], [[0.,0.,0.]]
obj = 10
for i in xrange(obj):
	r.append(np.array([rand(7.,15.,1,time.time())[0], 0., 0.]))
	v.append(np.array([0., rand(0.005,0.009, 1,time.time())[0], rand(0.0009,0.002, 1,time.time())[0]]))
	m.append(rand(0.0001,0.0003,1,time.time())[0])

lf = leapfrog.Leapfrog(m = m, r = r, v = v)

# lf.sum_(m = lf.m, excp = 0, r_i = lf.r[0], r_j = lf.r)

stop = 10000/2.#365. * 86400
step = 20#24.*3600
lf.integrate(time_step = step, time_stop = stop)


# print lf.time_evolution
vis.visualize(data = lf.time_evolution)
