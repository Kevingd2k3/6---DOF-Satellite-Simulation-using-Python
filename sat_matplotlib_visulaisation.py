import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R

# --- 1. CONFIGURATION ---
class Config:
    MU = 3.986004418e14      # Gravity Constant
    R_EARTH = 6371000.0      # Earth Radius (m)
    RHO_0 = 1.225            # Air density
    H_SCALE = 8500.0         # Scale height
    
    # EARTH ROTATION SPEED
    # Real Earth rotates once every 23h 56m 04s (Sidereal Day)
    # Omega = 2*pi / 86164 s ~= 7.2921e-5 rad/s
    OMEGA_EARTH = 7.2921159e-5 

    # SIMULATION SETTINGS
    # 400km Orbit Period is ~92.5 mins (5550 seconds)
    # We want 3 full loops => 3 * 5550 = ~16650 seconds.
    # Let's do 18000s (5 hours) to be safe.
    DURATION = 86400.0 

# --- 2. PHYSICS ENGINE ---
class SatelliteSim:
    def __init__(self):
        self.sat_mass = 50.0  
        self.sat_inertia = np.diag([5.0, 5.0, 4.0])
        self.sat_inv_inertia = np.linalg.inv(self.sat_inertia)
        self.cop_offset = np.array([0.1, 0.05, -0.05])
        self.cd = 2.2
        self.area = 1.0

    def dynamics(self, t, state):
        r = state[0:3]
        v = state[3:6]
        q = state[6:10]
        w = state[10:13]
        
        q = q / np.linalg.norm(q)
        r_mag = np.linalg.norm(r)
        
        # 1. Gravity (Force)
        a_grav = -Config.MU * r / (r_mag**3)
        f_grav = a_grav * self.sat_mass 

        # 2. Atmosphere
        alt = r_mag - Config.R_EARTH
        rho = Config.RHO_0 * np.exp(-alt/Config.H_SCALE) if alt > 0 else 0
        v_rel = v # Simplified (ignoring atmosphere rotation wind for now)
        v_mag = np.linalg.norm(v_rel)
        
        f_drag = np.zeros(3)
        t_drag = np.zeros(3)
        
        if v_mag > 0:
            f_drag = -0.5 * rho * v_mag * self.cd * self.area * v_rel
            # Drag Torque
            rot = R.from_quat(q)
            f_drag_body = rot.inv().apply(f_drag)
            t_drag = np.cross(self.cop_offset, f_drag_body)

        # 3. Gravity Gradient Torque
        rot = R.from_quat(q)
        r_body = rot.inv().apply(r)
        t_gg = (3*Config.MU/(r_mag**5)) * np.cross(r_body, self.sat_inertia @ r_body)

        # 4. Nadir Controller
        z_des = -r / r_mag
        h = np.cross(r, v)
        y_des = -h / np.linalg.norm(h)
        x_des = np.cross(y_des, z_des)
        
        target_rot = R.from_matrix(np.column_stack((x_des, y_des, z_des)))
        curr_rot = R.from_quat(q)
        q_err = (target_rot.inv() * curr_rot).as_quat()
        
        if q_err[3] < 0: q_err = -q_err
        t_ctrl = -1.0 * q_err[:3] - 5.0 * w 

        # Integration
        accel = (f_grav + f_drag) / self.sat_mass
        w_dot = self.sat_inv_inertia @ (t_gg + t_drag + t_ctrl - np.cross(w, self.sat_inertia @ w))
        
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
    print("Initializing VLEO Simulation...")
    sim = SatelliteSim()
    
    # 400km Orbit
    r0 = [Config.R_EARTH + 400000, 0, 0]
    
    # Accurate Velocity for 400km VLEO
    # v = sqrt(GM / r)
    v_circ = np.sqrt(Config.MU / r0[0])
    v0 = [0, v_circ, 0] # 90 degree inclination (Polar Orbit) to see Earth turn!
    
    q0 = [0, 0, 0, 1]
    w0 = [0.01, 0.01, 0.01]
    y0 = np.concatenate((r0, v0, q0, w0))
    
    # Create 400 frames for smooth 10-15s animation
    t_eval = np.linspace(0, Config.DURATION, 400) 
    
    print(f"Simulating  Orbits ({Config.DURATION/3600:.1f} hours)...")
    sol = solve_ivp(sim.dynamics, [0, Config.DURATION], y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

    print("Starting Animation...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- EARTH PRE-CALCULATION ---
    # We create the sphere points ONCE, then rotate them in the loop
    u, v = np.mgrid[0:2*np.pi:18j, 0:np.pi:9j] # Lower resolution for speed
    base_x = Config.R_EARTH * np.cos(u)*np.sin(v)
    base_y = Config.R_EARTH * np.sin(u)*np.sin(v)
    base_z = Config.R_EARTH * np.cos(v)

    # Plot Elements
    sat_dot, = ax.plot([], [], [], 'ro', markersize=8, label='Satellite')
    trail_line, = ax.plot([], [], [], 'y-', linewidth=1, label='Orbit Track')
    
    # A placeholder for the Earth wireframe plot
    earth_plot = None
    
    # Axis Limits
    limit = Config.R_EARTH * 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect([1,1,1])
    ax.legend()
    
    trail_x, trail_y, trail_z = [], [], []

    # Animation Loop
    for i in range(len(sol.t)):
        t_curr = sol.t[i]
        r_now = sol.y[0:3, i]
        
        # 1. Update Title
        ax.set_title(f"Time: {t_curr:.0f}s ({t_curr/3600:.1f}h) | Loops: {t_curr/5550:.1f}")
        
        # 2. ROTATE EARTH
        # Angle = Omega * Time
        theta = Config.OMEGA_EARTH * t_curr
        
        # Rotation Matrix (Rotation around Z-axis)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Manually rotate the base sphere coordinates
        # x_new = x*cos - y*sin
        # y_new = x*sin + y*cos
        rot_x = base_x * cos_t - base_y * sin_t
        rot_y = base_x * sin_t + base_y * cos_t
        rot_z = base_z # Z doesn't change
        
        # Redraw Earth
        if earth_plot:
            earth_plot.remove()
        
        earth_plot = ax.plot_wireframe(rot_x, rot_y, rot_z, color="b", alpha=0.15)
        
        # 3. Update Satellite
        sat_dot.set_data([r_now[0]], [r_now[1]])      
        sat_dot.set_3d_properties([r_now[2]])         
        
        # 4. Update Trail
        trail_x.append(r_now[0])
        trail_y.append(r_now[1])
        trail_z.append(r_now[2])
        trail_line.set_data(trail_x, trail_y)
        trail_line.set_3d_properties(trail_z)
        
        # Pause for animation speed
        plt.pause(0.001)

    plt.show()

if __name__ == "__main__":
    run()