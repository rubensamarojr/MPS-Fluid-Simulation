"""
Code largely inspired by:
https://www.youtube.com/watch?v=-0m05gzk8nk
https://doi.org/10.1016/C2016-0-03952-9
"""

import numpy as np
from sklearn import neighbors
from tqdm import tqdm
import pyvista as pv
import os
import shutil

# OUTPUT FOLDER NAME
FOLDER_NAME = "output_folder_1"

# NUMERICAL INPUTS
DIM = 2
DX = 0.025
DT = 0.0005
OUTPUT_INTERVAL = 20
RePND = 2.1 * DX
ReGrad = 2.1 * DX
ReDiv = 2.1 * DX
ReLap = 3.1 * DX
COLLISION_DISTANCE = 0.9 * DX
BETA = 0.97
COEFFICIENT_OF_RESTITUTION = 0.2
EPS = 0.01 * DX
GHOST = -1
FLUID = 0
WALL = 2
DUMMY_WALL = 3
SURFACE_PARTICLE = 1
INNER_PARTICLE = 0

# PHYSICAL INPUTS
DOMAIN_X_LIM = 1.0
DOMAIN_Y_LIM = 0.6
FLUID_X_LIM = 0.25
FLUID_Y_LIM = 0.5
ARRAY_SIZE = 5000
FINISH_TIME = 2.0
KINEMATIC_VISCOSITY = 1.0E-6
FLUID_DENSITY = 1000.0
SOUND_SPEED = 20.0
GAMMA_TAIT = 7
G_X = 0.0
G_Y = -9.8
G_Z = 0.0

def initDomain():
	global Position, Velocity, VelocityAux, Acceleration, Press, pndStar, BC, Pmin
	global ParticleType, NumberOfParticles, NumberOfFluidParticles, NumberOfGhostParticles

	nX = int(DOMAIN_X_LIM / DX) + 5
	nY = int(DOMAIN_Y_LIM / DX) + 5
	PartType0 = np.zeros(nX * nY, dtype=int)
	Pos0 = np.zeros(nX * nY * 3)
	Vel0 = np.zeros(nX * nY * 3)
	Press0 = np.zeros(nX * nY * 1)

	i = 0
	ff = 0
	for iX in range(-4, nX):
		for iY in range(-4, nY):
			x = DX * iX
			y = DX * iY
			z = 0.0
			flagOfParticleGeneration = False

			# dummy wall region
			if -4.0 * DX + EPS < x <= DOMAIN_X_LIM + 4.0 * DX + EPS and \
					0.0 - 4.0 * DX + EPS < y <= DOMAIN_Y_LIM + EPS:
				PartType0[i] = DUMMY_WALL
				flagOfParticleGeneration = True

			# wall region
			if -2.0 * DX + EPS < x <= DOMAIN_X_LIM + 2.0 * DX + EPS and \
					0.0 - 2.0 * DX + EPS < y <= DOMAIN_Y_LIM + EPS:
				PartType0[i] = WALL
				flagOfParticleGeneration = True

			# wall region
			if -4.0 * DX + EPS < x <= DOMAIN_X_LIM + 4.0 * DX + EPS and \
					DOMAIN_Y_LIM - 2.0 * DX + EPS < y <= DOMAIN_Y_LIM + EPS:
				PartType0[i] = WALL
				flagOfParticleGeneration = True

			# empty region
			if 0.0 + EPS < x <= DOMAIN_X_LIM + EPS and y > 0.0 + EPS:
				flagOfParticleGeneration = False

			# fluid region
			if 0.0 + EPS < x <= FLUID_X_LIM + EPS and 0.0 + EPS < y <= FLUID_Y_LIM + EPS:
				PartType0[i] = FLUID
				flagOfParticleGeneration = True
				ff += 1

			if flagOfParticleGeneration:
				Pos0[i * 3] = x
				Pos0[i * 3 + 1] = y
				Pos0[i * 3 + 2] = z
				Press0[i] = 0.0
				if i == 0:
					Position = np.array([[Pos0[i * 3 + 0], Pos0[i * 3 + 1]],])
					ParticleType = np.array([[PartType0[i]],])
					Press = np.array([[Press0[i]],])
				else:
					PositionAux = np.array([[Pos0[i * 3 + 0], Pos0[i * 3 + 1]],])
					Position = np.concatenate((Position, PositionAux), axis=0)
					ParticleTypeAux = np.array([[PartType0[i]],])
					ParticleType = np.concatenate((ParticleType, ParticleTypeAux), axis=0)
					PressAux = np.array([[Press0[i]],])
					Press = np.concatenate((Press, PressAux), axis=0)
				i += 1

	del Pos0, PartType0, Press0

	NumberOfParticles = i
	NumberOfFluidParticles = ff
	NumberOfGhostParticles = 0
	
	Velocity = np.zeros_like(Position)
	VelocityAux = np.zeros_like(Position)
	Acceleration = np.zeros_like(Position)
	pndStar = np.zeros_like(Press)
	BC = np.zeros_like(ParticleType)
	Pmin = np.zeros_like(Press)

# More info about weight function in MPS, see Figs 4-6 in https://doi.org/10.1016/j.enganabound.2021.06.018
def weight(distance, re):
	# Rational weight (singular)
	if distance < re:
		wij = (re / distance) - 1.0
	else:
		wij = 0.0
	return wij
	# 2nd order polynomial  (non-singular)
	# if distance < re:
	# 	wij = ((distance / re) - 1.0) * ((distance / re) - 1.0)
	# else:
	# 	wij = 0.0
	# return wij

def constantParameters():
	global Re2PND, Re2Grad, Re2Lap, pnd0, pnd0Grad, pnd0Div, pnd0Lap, Lambda0
	global collisionDistance2, FileNumber, Time, lapC1, pressC1, gradC1, colC1, viscC1, divC1

	Re2PND = RePND * RePND
	Re2Grad = ReGrad * ReGrad
	Re2Lap = ReLap * ReLap
	pnd0, pnd0Grad, pnd0Div, pnd0Lap, Lambda0 = pnd0Lambda0()
	
	collisionDistance2 = COLLISION_DISTANCE * COLLISION_DISTANCE
	lapC1 = KINEMATIC_VISCOSITY*2.0*DIM/(pnd0Lap*Lambda0)
	pressC1 = FLUID_DENSITY * SOUND_SPEED**2 / GAMMA_TAIT
	gradC1 = DIM/(pnd0Grad*FLUID_DENSITY)
	colC1 = 1.0 + COEFFICIENT_OF_RESTITUTION
	viscC1 = 2.0 * DIM / (pnd0Lap * Lambda0 * FLUID_DENSITY)
	divC1 = DIM / pnd0Div

def pnd0Lambda0():
	iZ_start = 0
	iZ_end = 1
	pnd0 = 0.0
	pnd0Grad = 0.0
	pnd0Div = 0.0
	pnd0Lap = 0.0
	Lambda0 = 0.0
	xi = 0.0
	yi = 0.0
	zi = 0.0

	for iX in range(-4, 5):
		for iY in range(-4, 5):
			for iZ in range(iZ_start, iZ_end):
				if iX == 0 and iY == 0 and iZ == 0:
					continue
				xj = DX * iX
				yj = DX * iY
				zj = DX * iZ
				distance2 = (xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2
				distance = np.sqrt(distance2)
				pnd0 += weight(distance, RePND)
				pnd0Grad += weight(distance, ReGrad)
				pnd0Div += weight(distance, ReDiv)
				pnd0Lap += weight(distance, ReLap)
				Lambda0 += distance2 * weight(distance, ReLap)

	Lambda0 = Lambda0 / pnd0Lap

	return pnd0, pnd0Grad, pnd0Div, pnd0Lap, Lambda0

def neighbourSearch():
	global neighbor_ids, distances
	neighbor_ids, distances = neighbors.KDTree(Position,).query_radius(Position,ReLap,return_distance=True,sort_results=True,)

def calcLapVel():
	for i in range(NumberOfParticles):
		if ParticleType[i] == FLUID:
			result = 0.0
			for j_in_list, j in enumerate(neighbor_ids[i]):
				if j != i:
					rijMag = distances[i][j_in_list]
					result += (Velocity[j] - Velocity[i]) * weight(rijMag, ReLap)
			Acceleration[i] = result * lapC1

def calcForces():
	for i in range(NumberOfParticles):
		Acceleration[i] = Acceleration[i] + (G_X, G_Y)

def predVelPos():
	for i in range(NumberOfParticles):
		if ParticleType[i] == FLUID:
			Velocity[i] += Acceleration[i] * DT
			Position[i] += Velocity[i] * DT

# Mor info about collision, see item 3.6 in https://doi.org/10.1016/j.cma.2010.12.001 
def checkCollision():
	for i in range(NumberOfParticles):
		VelocityAux[i] = 0.0
		if ParticleType[i] == FLUID:
			velAux = Velocity[i]
			fDT = 0.0
			for j_in_list, j in enumerate(neighbor_ids[i]):
				if j != i:
					rij = Position[j] - Position[i]
					rijMag2 = np.dot(rij, rij)
					if rijMag2 < collisionDistance2:
						rijMag = np.sqrt(rijMag2)
						vij = Velocity[i] - Velocity[j]
						fDT = np.dot(vij, rij)
						if fDT > 0.0:
							fDT *= 0.5 * colC1 / (rijMag * rijMag)
							velAux -= rij * fDT
			VelocityAux[i] = velAux

	for i in range(NumberOfParticles):
		Velocity[i] = VelocityAux[i]

def calcPND():
	for i in range(NumberOfParticles):
		if ParticleType[i] != GHOST:
			result = 0.0
			for j_in_list, j in enumerate(neighbor_ids[i]):
				if j != i:
					rij = Position[j] - Position[i]
					rijMag = np.sqrt(np.dot(rij, rij))
					result += weight(rijMag, RePND)
			pndStar[i] = result

# More info about the PND below, see item 2.1.3 in https://doi.org/10.1002/fld.5083
def calcDeltaPND():
	for i in range(NumberOfParticles):
		if ParticleType[i] != GHOST:
			pndi = pndStar[i]
			Di = 0.0
			DivV = 0.0
			result = 0.0
			for j_in_list, j in enumerate(neighbor_ids[i]):
				if j != i:
					rij = Position[j] - Position[i]
					rijMag = np.sqrt(np.dot(rij, rij))
					if ParticleType[i] == FLUID and ParticleType[j] == FLUID:
						pgh = - FLUID_DENSITY * np.dot((G_X, G_Y), rij)
						Di += DT * viscC1 * (Press[j] - Press[i] - pgh) * weight(rijMag, ReGrad)
					if pndi > 0.0:
						vij = Velocity[j] - Velocity[i]
						DivV += divC1 * np.dot(vij, rij) * (pndStar[j] / pndi) * weight(rijMag, ReDiv) / (rijMag * rijMag);
						
			pndStar[i] = pndStar[i]*(1.0+DT*(Di-DivV));

def setBC():
	for i in range(NumberOfParticles):
		if pndStar[i] < (BETA * pnd0):
			BC[i] = SURFACE_PARTICLE
		else:
			BC[i] = INNER_PARTICLE

def calcPressure():
	for i in range(NumberOfParticles):
		Press[i] = 0.0
		if ParticleType[i] != GHOST:
			Press[i] = pressC1 * (pow(pndStar[i]/pnd0, GAMMA_TAIT) - 1.0)
			if Press[i] < 0.0 or BC[i] == SURFACE_PARTICLE:
				Press[i] = 0.0

def extrapolateWallPressure():
	for i in range(NumberOfParticles):
		if ParticleType[i] == WALL or ParticleType[i] == DUMMY_WALL:
			pressi = 0.0
			ni = 0.0
			for j_in_list, j in enumerate(neighbor_ids[i]):
				if j != i:
					if ParticleType[j] == FLUID:
						rij = Position[j] - Position[i]
						pgh = - FLUID_DENSITY * np.dot((G_X, G_Y), rij)
						rijMag = np.sqrt(np.dot(rij, rij))
						wij = weight(rijMag, RePND)
						pressi += (Press[j] - pgh) * wij
						ni += wij
				
				if Press[i] < 0.0 or BC[i] == SURFACE_PARTICLE:
					Press[i] = 0.0
				else:
					if ni > 0.0:
						Press[i] = pressi/ni;

def minimumPressure():
	for i in range(NumberOfParticles):
		if ParticleType[i] == FLUID:
			result = Press[i]
			for j_in_list, j in enumerate(neighbor_ids[i]):
				if j != i:
					rij = Position[j] - Position[i]
					rijMag = np.sqrt(np.dot(rij, rij))
					if rijMag < ReGrad:
						if Press[j] < result:
							result = Press[j]
			Pmin[i] = result

def calcGradPress(gradType):
	for i in range(NumberOfParticles):
		if ParticleType[i] == FLUID:
			result = 0.0
			for j_in_list, j in enumerate(neighbor_ids[i]):
				if j != i:
					rij = Position[j] - Position[i]
					rijMag = np.sqrt(np.dot(rij, rij))
					if rijMag < ReGrad:
						if gradType == 0:
							result += (Press[j] - Press[i]) * (Position[j] - Position[i]) * weight(rijMag,ReGrad) / (rijMag * rijMag)
						elif gradType == 1:
							result += (Press[j] - Pmin[i]) * (Position[j] - Position[i]) * weight(rijMag,ReGrad) / (rijMag * rijMag)
						elif gradType == 2:
							result += (Press[j] + Press[i]) * (Position[j] - Position[i]) * weight(rijMag,ReGrad) / (rijMag * rijMag)
			Acceleration[i] = -gradC1*result

def corrVelPos():
	for i in range(NumberOfParticles):
		if ParticleType[i] == FLUID:
			Velocity[i] += Acceleration[i] * DT
			Position[i] += Acceleration[i] * DT * DT
		Acceleration[i] = 0.0

def checkGhost():
	global NumberOfGhostParticles
	numOfGhosts = 0
	for i in range(NumberOfParticles):
		if ParticleType[i] == FLUID:
			if Position[i,0] > (DOMAIN_X_LIM + 4 * DX) or Position[i,0] < -3.0 * DX or Position[i,1] > (DOMAIN_Y_LIM + 4 * DX) or Position[i,1] < -3.0 * DX:
				ParticleType[i] = GHOST
				Position[i] = (DOMAIN_X_LIM + 8 * DX, DOMAIN_Y_LIM + 8 * DX)
				Velocity[i] = (0.0, 0.0)
				numOfGhosts += 1

	NumberOfGhostParticles += numOfGhosts

def printOUT(iterOUT):
	# 2D -> 3D
	pointsAux = np.ones((NumberOfParticles, 1))*0.0
	Position3D = np.append(Position, pointsAux, axis=1)
	Velocity3D = np.append(Velocity, pointsAux, axis=1)

	# Save initial configuration as a VTK file
	data = pv.PolyData(Position3D)
	data.point_data['velocity'] = Velocity3D
	data.point_data['press'] = Press
	data.point_data['pnd'] = pndStar
	data.point_data['BC'] = BC
	data.point_data['type'] = ParticleType
	vtk_file = FOLDER_NAME + f"/step_{iterOUT}.vtk"
	data.save(vtk_file)

def main():

	# Create output folder
	path = "./" + FOLDER_NAME
	# checking if the directory demo_folder exist or not.
	if os.path.exists(path):
		# Clear directory
		shutil.rmtree(path)
		# then create it
		os.makedirs(path)
	else:
		# if the demo_folder directory is not present 
		# then create it.
		os.makedirs(path)

	initDomain()
	constantParameters()
	neighbourSearch()
	calcPND()
	setBC()
	iOUT = 0
	printOUT(iOUT)

	nSteps = int(FINISH_TIME/DT)
	## Loop for all steps
	for iter in tqdm(range(nSteps)):
		neighbourSearch()
		calcLapVel()
		calcForces()
		predVelPos()
		checkCollision()
		calcPND()
		#calcDeltaPND() # (Unstable in this implementation!)
		setBC()
		calcPressure()
		extrapolateWallPressure()
		minimumPressure()
		calcGradPress(1) # 0: Pj-Pi (Unstable!), 1: Pj-Pmin, 2: Pj+Pi
		corrVelPos()
		checkGhost()
		if (iter % OUTPUT_INTERVAL) == 0:
			iOUT = iOUT + 1
			printOUT(iOUT)
		
		if NumberOfGhostParticles > 0.3 * NumberOfFluidParticles:
			print("Number of ghost particles exceeds 30% of fluid particles")
			break

if __name__ == "__main__":
	main()