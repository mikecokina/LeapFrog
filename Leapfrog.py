#!/usr/bin/python
import numpy as np
import scipy as sp
from scipy import constants
import sys

class Leapfrog:
	def __init__(self, m = None, r = None, v = None):
		if len(m) != len(r) or len(r) != len(v):	print "DimensionError:"
		self.m = np.array(m)
		self.r = [np.array(item) for item in r]
		self.v = [np.array(item) for item in v]
		self.points = len(self.m)
		
		self.time_evolution = []
		
		# constants
		# self.G = sp.constants.G
		self.G = 0.017202098950**2
		self.au = sp.constants.au
		
		
	def distance(self, v1, v2):
		return np.linalg.norm(np.array(v1)-np.array(v2))
		
		
	def v_size(self, v):
		return np.sqrt(np.dot(np.array(v), np.array(v)))
		
		
	def total_energy(self, m = [], r = [], v = []):
		sum_1, sum_2 = 0., 0.
		
		for i in range(0, v):
			sum_1 += m[i] * (v[i][0]**2 + v[i][1]**2 + v[i][2]**2)
		
		for i in range(0, len(v) - 1):
			for j in range(i + 1, len(v)):
				sum_2 += ( m[i] * m[j] ) / ( distance(r[i], r[j]) )
				
		T, U = .5 * sum_1, - self.G * sum_2
		return T + U
		
		
	def sum_(self, m = None, excp = 0, r_i = None, r_j = None):
		sum = np.array([0., 0., 0.])
		for j in range(0, len(r_j)):
			if j == excp:	continue
			else:
				sum += m[j] * (r_i - r_j[j]) / ( self.distance(r_i, r_j[j]) **3 )
		return sum
		
		
	def integrate(self, time_step = 10, time_stop = 100):
	
		v_overline, r_actual, v_actual = [], [], []
		
		for _ in range(0, self.points):
			v_overline.append(np.array([0., 0., 0.]))
			r_actual.append(np.array([0., 0., 0.]))
			v_actual.append(np.array([0., 0., 0.]))
	
		m_zero = self.m
		r_zero = self.r
		v_zero = self.v
		
		time_stamp = 0.0
		time_evolution = []
		
		while time_stamp < time_stop:
			to_append = r_zero[:]
			time_evolution.append(to_append)
			
			for i in range(0, self.points):
				pre_factor = self.G * (time_step / 2.)
				
				# overline velocity
				sum_overline = self.sum_(m = m_zero, excp = i, r_i = r_zero[i], r_j = r_zero)
				v_overline[i] = v_zero[i] - ( sum_overline * pre_factor )
				
				# actual position
				r_actual[i] = r_zero[i] + ( v_overline[i] * time_step )
				
			for i in range(0, self.points):
				# actual velocity
				sum_current = self.sum_(m = m_zero, excp = i, r_i = r_actual[i], r_j = r_actual)
				v_actual[i] = v_overline[i] - ( sum_current * pre_factor )
				
			for i in range(0, self.points):
				r_zero[i], v_zero[i] = r_actual[i], v_actual[i]
			
			
			time_stamp += time_step
		
		self.time_evolution = time_evolution[:]

