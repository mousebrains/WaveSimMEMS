# Simulate ocean waves with for a glider on the surface and generate MEMs sensor responses

A Python3 script which needs Pandas and NumPy installed.

To run the script:<br/>
_./simWaves.py exp.00.yml_
<br/>
which generates two files:
* exp.00.log which is a copy of the log output
* exp.00.csv which is the motion of a water parcel

Currently the csv has the following columns:
* t time in seconds starting at 0
* x, y, and z are the water parcel's position in three space with z being vertical and y true north
* dxdt, dydt, and dzdt, are the linear velocities in m/sec
* dxydt, dxzdt, and dyzdt are the angular velocities in radians/sec

Items still to be done include:
* add a wave to glider transfer function
* generate accelerations from velocities
* generate angular accelerations from angular velocities
* add sensor noise terms to linear and gyro simulated observations
* update README.md with better information
