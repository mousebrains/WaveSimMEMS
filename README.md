# Simulate ocean waves with for a glider on the surface and generate MEMs sensor responses

A Python3 script which needs Pandas and NumPy installed.

To run the script:<br/>
_./simWaves.py exp.00.yml_
<br/>
which generates multipe files:
* exp.00.log which is a copy of the log output
* exp.00.csv which is the motion of a water parcel summed for all waves
* exp.00.*.info.csv are the individual wave trains
* exp.00.*.ellipse.csv are the observations for the individual wave trains

## The main csv has the following columns:
* t time in seconds starting at 0
* [x, y, z] are the water parcel's position in meters in the true north earth coordinates, x is eastward, y northward, and z vertically
* v[x,y,z] are the instantaneous linear velocities in m/sec
* a[x,y,z] are the instantaneous linear accelerations in m/sec^2
* omega[x,y,z] are the instantaneous angular velocities in radians/sec
* alpha[x,y,z] are the instantaneous angular accelerations in radians/sec^2

## The info csv file columns are:
* period wave's period in seconds
* amp wave amplitude in m, this is half the peak to trough height
* hdg wave's direction from true north in degrees
* lambda wavelength in m
* spd wave phase speed in m/sec
* a and b are ellipse semi major and minor axis parameters in m

## The ellipse csv has the following columns:
* t time in seconds starting at 0
* w[x, z] are the water parcel's position in meters in the wave's coordinate system, x is along the direction of motion
* wv[x,z] are the instantaneous linear velocities in m/sec in the wave's coordinate system
* wa[x,z] are the instantaneous linear accelerations in m/sec^2 in the wave's coordinate system
* wOmega is the instantaneous angular velocity in the wave's y direction in radians/sec
* wAlpha is the instantaneous angular acceleration in the wave's y direction in radians/sec^2
* [x, y, z] are the water parcel's position in meters in the true north earth coordinates, x is eastward, y northward, and z vertically
* v[x,y,z] are the instantaneous linear velocities in m/sec
* a[x,y,z] are the instantaneous linear accelerations in m/sec^2
* omega[x,y,z] are the instantaneous angular velocities in radians/sec
* alpha[x,y,z] are the instantaneous angular accelerations in radians/sec^2

# Items still to be done include:
* add a wave to glider transfer function
* add sensor noise terms to linear and gyro simulated observations
* update README.md with better information

# Algorithm:

The YAML file specifies all the parameters needed for a simulation

For each wave specified, a wave train is generated which is a set of amplitudes and periods such that the cumulative period time exceeds the simulation duration.

For each wave in the train, a depth dependent wavelength and phase speed are calculated, along with the particle orbit ellipse semi minor and major axis parameters.

A set of times to sample at is generated.

For each wave train the following is done:
* A random initial phase is calculated so the wave trains don't all start at zero.
* In the wave's coordinate system, positions, velocities, accelerations, angular velocity, and angular acceleration are calculated.
* The above positions, velocities, and accelerations are rotated into the earth's true north coordinate system.

The individual wave trains are summed in the earth's coordinate system.

## To be added

* add glider response to the wave accelerations/velocities.
* estimate sensor response, i.e. time averaged accelerations instead of instantaneous.
* add noise to the sensor readings
