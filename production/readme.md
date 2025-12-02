Production runs on HPC:

Python script is used to construct nanocrystal cell from asymmetric unit + symmetry operations -> primitive cell + replication using lattice vectors -> supercell/nanocrystal.

Nanocrystals are 4x4x3 primitive cells for monoclinic run and 3x4x2 primitive for orthorhombic run to produce an approximate cube of length 30 Angstroms.

Supercells are produced by removing tilt from nanocrystal cells and padding each axes by 30 Angstroms on each side (total 60 Angstroms per axes).

Number of water molecules are computed by taking into account the density of water and the additional volume created around the nanocrystal by padding.

Water molecules are created one by one at random points in the grid avoiding locations too close to other atoms using overlap 1.5.

Temperature and pressure set to 310K (37 Celsius) as human body temperature and 1 atm.

Simulation settings are 2fs timestep for 2 500 000 tmesteps (5 ns). Thermo and trajectory info every 2500 steps (5 ps).

Google drive link for trajectory files/data:
https://drive.google.com/drive/folders/1E85vHUdiNx1qJykNDcwLEFsQ3wYoFdIJ?usp=sharing

Thermo data is embedded in .out files.
