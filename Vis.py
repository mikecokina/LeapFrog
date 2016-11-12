# visualisation
#!/usr/bin/python
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def update(num, plots, lines, x, y, z):
	for plot, line, X, Y, Z in zip(plots, lines, x, y, z):
		line.set_data(np.array([X[:num], Y[:num]]))
		line.set_3d_properties(np.array(Z[:num]))
		if num != 0:
			plot.set_data(np.array([X[num-1:num], Y[num-1:num]]))
			plot.set_3d_properties(np.array(Z[num-1:num]))
		else:
			plot.set_data(np.array([X[:num], Y[:num]]))
			plot.set_3d_properties(np.array(Z[:num]))
	return plots, lines
	
	
def visualize(data):
	fig = plt.figure()
	ax = p3.Axes3D(fig)
	
	
	# re-format input data
	x, y, z = [[] for i in range(0, len(data[0]))], [[] for i in range(0, len(data[0]))], [[] for i in range(0, len(data[0]))]
	for i in range(0, len(data)):
		for j in range(0, len(data[i])):
			x[j].append(data[i][j][0])
			y[j].append(data[i][j][1])
			z[j].append(data[i][j][2])
			
	# x = [[3,4,5,6,7,8,9],[3,4,5,6,7,7,8]]
	# y = [[3,4,6,8,9,1,3],[4,5,7,8,9,0,1]]
	# z = [[4,5,3,4,6,7,8],[4,5,4,6,7,7,8]]
	plots = [ax.plot(X[0:1], Y[0:1], Z[0:1], "o")[0] for X, Y, Z in zip(x, y, z)]
	lines = [ax.plot(X[0:1], Y[0:1], Z[0:1])[0] for X, Y, Z in zip(x, y, z)]
	
	maxX, maxY, maxZ = x[0][0], y[0][0], z[0][0]
	for i in xrange(0, len(x)):
		for j in xrange(0, len(x[i])):
			if x[i][j] > maxX: maxX = np.ceil(x[i][j])
			if y[i][j] > maxY: maxY = np.ceil(y[i][j])
			if z[i][j] > maxZ: maxZ = np.ceil(z[i][j])
		
	minX, minY, minZ = x[0][0], y[0][0], z[0][0]
	for i in xrange(0, len(x)):
		for j in xrange(0, len(x[i])):
			if x[i][j] < minX: minX = np.floor(x[i][j])
			if y[i][j] < minY: minY = np.floor(y[i][j])
			if z[i][j] < minZ: minZ = np.floor(z[i][j])
	

	
	
	# Setting the axes properties
	ax.set_xlim3d([minX, maxX])
	ax.set_xlabel('X')

	ax.set_ylim3d([minY, maxY])
	ax.set_ylabel('Y')

	ax.set_zlim3d([minZ, maxZ])
	ax.set_zlabel('Z')
	
	ani = animation.FuncAnimation(fig, update, fargs=(plots, lines, x, y, z), interval=1)
	
	plt.show()
	
	
