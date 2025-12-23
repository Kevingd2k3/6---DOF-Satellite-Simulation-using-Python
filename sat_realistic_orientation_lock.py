import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R

# --- 1. CONFIGURATION ---
class Config:
    MU = 3.986004418e14       # Gravity Constant (m^3/s^2)
    R_EARTH = 6371000.0       # Earth Radius (m)
    RHO_0 = 1.225             # Air Density at Sea Level (kg/m^3)
    H_SCALE = 8500.0          # Atmosphere Scale Height (m)
    OMEGA_EARTH = 7.2921e-5   # Earth Rotation Speed (rad/s)
    
    # Run for 3 Orbits (approx 5.5 hours)
    DURATION = 20000.0 

# --- 2. PHYSICS ENGINE WITH "SPICE" ---
class SatelliteSim:
    def __init__(self):
        self.sat_mass = 50.0  
        self.sat_inertia = np.diag([5.0, 5.0, 4.0])
        self.sat_inv_inertia = np.linalg.inv(self.sat_inertia)
        
        # THE SPICE: Center of Pressure Offset
        # Drag hits this point, creating a twisting torque
        self.cop_offset = np.array([0.1, 0.05, -0.05])
        self.cd = 2.2
        self.area = 1.0

    def dynamics(self, t, state):
        # Unpack State
        r = state[0:3]
        v = state[3:6]
        q = state[6:10] # Quaternion [x,y,z,w]
        w = state[10:13]
        
        q = q / np.linalg.norm(q)
        r_mag = np.linalg.norm(r)
        
        # 1. Gravity Force (F = ma * mass)
        a_grav = -Config.MU * r / (r_mag**3)
        f_grav = a_grav * self.sat_mass 

        # 2. VLEO Atmosphere Model
        alt = r_mag - Config.R_EARTH
        rho = Config.RHO_0 * np.exp(-alt/Config.H_SCALE) if alt > 0 else 0
        v_rel = v 
        v_mag = np.linalg.norm(v_rel)
        
        f_drag = np.zeros(3)
        t_drag = np.zeros(3)
        
        if v_mag > 0:
            # Drag Force
            f_drag = -0.5 * rho * v_mag * self.cd * self.area * v_rel
            
            # SPICE: Calculate Drag Torque
            # We must rotate the drag force into the Body Frame to cross it with the offset
            rot = R.from_quat(q)
            f_drag_body = rot.inv().apply(f_drag)
            t_drag = np.cross(self.cop_offset, f_drag_body)

        # 3. Gravity Gradient Torque
        rot = R.from_quat(q)
        r_body = rot.inv().apply(r)
        t_gg = (3*Config.MU/(r_mag**5)) * np.cross(r_body, self.sat_inertia @ r_body)

        # 4. Nadir Controller (The Brain)
        # Calculate where we WANT to point (Z-axis to Earth)
        z_des = -r / r_mag
        h = np.cross(r, v)
        y_des = -h / np.linalg.norm(h)
        x_des = np.cross(y_des, z_des)
        
        target_rot = R.from_matrix(np.column_stack((x_des, y_des, z_des)))
        curr_rot = R.from_quat(q)
        q_err = (target_rot.inv() * curr_rot).as_quat()
        
        if q_err[3] < 0: q_err = -q_err
        
        # Gains: High gains to fight the thick VLEO atmosphere
        Kp = 2.0
        Kd = 10.0
        t_ctrl = -Kp * q_err[:3] - Kd * w 

        # 5. Equations of Motion
        accel = (f_grav + f_drag) / self.sat_mass
        w_dot = self.sat_inv_inertia @ (t_gg + t_drag + t_ctrl - np.cross(w, self.sat_inertia @ w))
        
        # Quaternion Derivative
        qx, qy, qz, qw = q
        wx, wy, wz = w
        q_dot = 0.5 * np.array([
            qw*wx + qy*wz - qz*wy,
            qw*wy - qx*wz + qz*wx,
            qw*wz + qx*wy - qy*wx,
            -qx*wx - qy*wy - qz*wz
        ])
        
        return np.concatenate((v, accel, q_dot, w_dot))

# --- 3. VISUALIZATION ---
def run():
    print("Initializing VLEO Simulation (With Attitude Arrows)...")
    sim = SatelliteSim()
    
    # 400km Orbit
    r0 = [Config.R_EARTH + 400000, 0, 0]
    v_circ = np.sqrt(Config.MU / r0[0])
    v0 = [0, v_circ, 0] # Polar Orbit
    
    q0 = [0, 0, 0, 1]
    w0 = [0.01, 0.01, 0.01]
    y0 = np.concatenate((r0, v0, q0, w0))
    
    # 500 frames for smooth playback
    t_eval = np.linspace(0, Config.DURATION, 500) 
    
    print("Solving Dynamics (This takes a moment due to 'Spice' physics)...")
    sol = solve_ivp(sim.dynamics, [0, Config.DURATION], y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

    print("Starting Animation...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Earth Pre-calc
    u, v = np.mgrid[0:2*np.pi:18j, 0:np.pi:9j]
    base_x = Config.R_EARTH * np.cos(u)*np.sin(v)
    base_y = Config.R_EARTH * np.sin(u)*np.sin(v)
    base_z = Config.R_EARTH * np.cos(v)

    # --- PLOT ELEMENTS ---
    # 1. Satellite (Dot)
    sat_dot, = ax.plot([], [], [], 'ko', markersize=4, label='Sat Center')
    
    # 2. Orbit Trail
    trail_line, = ax.plot([], [], [], 'y-', linewidth=1, label='Orbit')
    
    # 3. Attitude Arrows (The Visual Spice)
    # We will draw lines representing Body X, Y, Z axes
    # Scale arrows to be visible (2000 km long purely for visibility)
    arrow_len = 2000000.0 
    quiver_x, = ax.plot([], [], [], 'r-', linewidth=3, label='Body X (Forward)')
    quiver_z, = ax.plot([], [], [], 'b-', linewidth=3, label='Body Z (Nadir/Down)')
    
    earth_plot = None
    
    limit = Config.R_EARTH * 1.8
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect([1,1,1])
    ax.legend(loc='upper right')
    
    trail_x, trail_y, trail_z = [], [], []

    for i in range(len(sol.t)):
        t_curr = sol.t[i]
        r_now = sol.y[0:3, i]
        q_now = sol.y[6:10, i]
        
        # --- A. EARTH ROTATION ---
        theta = Config.OMEGA_EARTH * t_curr
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        rot_x = base_x * cos_t - base_y * sin_t
        rot_y = base_x * sin_t + base_y * cos_t
        
        if earth_plot: earth_plot.remove()
        earth_plot = ax.plot_wireframe(rot_x, rot_y, base_z, color="b", alpha=0.1)
        
        # --- B. UPDATE TRAIL ---
        trail_x.append(r_now[0]); trail_y.append(r_now[1]); trail_z.append(r_now[2])
        trail_line.set_data(trail_x, trail_y)
        trail_line.set_3d_properties(trail_z)
        
        # --- C. UPDATE SATELLITE POSITION ---
        sat_dot.set_data([r_now[0]], [r_now[1]])
        sat_dot.set_3d_properties([r_now[2]])
        
        # --- D. UPDATE ATTITUDE ARROWS (THE SPICE) ---
        # Get rotation matrix from quaternion
        rot_mat = R.from_quat(q_now).as_matrix()
        
        # Body X Axis (Red) - Where the satellite is "facing"
        # Transform vector [1,0,0] by rotation matrix
        vec_x = rot_mat @ np.array([1, 0, 0]) * arrow_len
        quiver_x.set_data([r_now[0], r_now[0]+vec_x[0]], [r_now[1], r_now[1]+vec_x[1]])
        quiver_x.set_3d_properties([r_now[2], r_now[2]+vec_x[2]])
        
        # Body Z Axis (Blue) - The Camera/Sensor (Should point to Earth)
        # Transform vector [0,0,1]
        vec_z = rot_mat @ np.array([0, 0, 1]) * arrow_len
        quiver_z.set_data([r_now[0], r_now[0]+vec_z[0]], [r_now[1], r_now[1]+vec_z[1]])
        quiver_z.set_3d_properties([r_now[2], r_now[2]+vec_z[2]])
        
        # --- TITLE INFO ---
        # Calculate Error Angle (How much is the drag winning?)
        # Ideal Z is -r / |r|
        ideal_z = -r_now / np.linalg.norm(r_now)
        actual_z = rot_mat @ np.array([0, 0, 1])
        # Dot product = cos(angle)
        dot_prod = np.clip(np.dot(ideal_z, actual_z), -1.0, 1.0)
        err_deg = np.degrees(np.arccos(dot_prod))
        
        ax.set_title(f"T={t_curr:.0f}s | Pointing Err after Drag Correction: {err_deg:.2f}°\n(Blue Line Satellite Camera Pointing To Earth Center)", fontsize=10)
        
        plt.pause(0.001)

    plt.show()

if __name__ == "__main__":
    run()