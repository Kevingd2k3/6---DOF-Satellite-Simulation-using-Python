# 6-DOF Satellite Simulation (VLEO) Using Python

![Language](https://img.shields.io/badge/Code-Python-blue) ![Library](https://img.shields.io/badge/Lib-Matplotlib-orange) ![Library](https://img.shields.io/badge/Lib-VTK-red) ![Library](https://img.shields.io/badge/Lib-SciPy-green) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

##  Project Overview

This project is a high-fidelity flight dynamics simulation modeling the **6-Degrees-of-Freedom (6-DOF)** motion of a satellite in a **Very Low Earth Orbit (VLEO)** altitude of 400 km. Unlike standard orbital propagators that only track position, this simulation couples **Orbital Mechanics** (Translation) with **Attitude Dynamics** (Rotation).

The core simulation focuses on the unique challenges of VLEO, specifically the "Spice": **Aerodynamic Disturbance Torques**. At 400 km, the atmosphere is thick enough that drag forces acting on the satellite's Center of Pressure (offset from the Center of Mass) create destabilizing torques. An active **Nadir-Pointing Control System** is implemented to fight these forces and keep the satellite's sensors locked on Earth.

##  Objectives

* **Simulate Realistic Physics:** Model the non-linear coupling between orbital velocity, atmospheric density, and aerodynamic drag.
* **Demonstrate VLEO Instability:** Visualize how a passive satellite tumbles due to drag torques without active control.
* **Implement Attitude Control:** Design and test a PD (Proportional-Derivative) controller to maintain a Nadir-pointing orientation.
* **Dual Visualization Pipeline:** Compare lightweight real-time plotting (Matplotlib) against high-fidelity 3D rendering (VTK).

## Future Scope
* **Solar Radiation Pressure (SRP):** Implementing disturbance torques from solar photons for higher orbits (GEO).
* **Reaction Wheel Saturation:** Simulating the physical limits of the actuators (momentum dumping).
* **Kalman Filtering:** Adding sensor noise to the simulation and implementing a filter to estimate true attitude.
* **Interplanetary Missions:** Adapting the physics engine for Martian or Lunar orbits.

## 📂 Project Structure & Modules

```text
6 - DOF SATELLITE SIMULATION USING PYTHON
├── README.md                           # Project documentation
├── Report.pdf                          # Detailed technical report of the physics math
├── sat.stl                             # 3D Stereolithography model of the satellite
├── sat_matplotlib_visulaisation.py     # MODULE 1: Fast, lightweight trajectory plotting
├── sat_realistic_orientation_lock.py   # MODULE 2: The "Spiced" physics engine with Earth rotation
├── sat_vtk_visualisation.py            # MODULE 3: High-fidelity graphics engine
├── VLEO_Sat_Sim_24Hr.mp4               # Demo: Long-duration simulation output
├── VLEO_Sat_Sim_w_OrientationLock.mp4  # Demo: Active control system in action
└── VTK Visualisation.png               # Screenshot of the VTK render window

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.