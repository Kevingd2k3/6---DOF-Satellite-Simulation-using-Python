import vtk
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R
import time

# --- 1. CONFIGURATION (Strict SI Units: Meters, Seconds, kg) ---
class Config:
    MU = 3.986004418e14     # Standard Gravitational Parameter (m^3/s^2)
    R_EARTH = 6371000.0     # Earth Radius (m)
    J2 = 1.08263e-3         # Oblateness factor
    
    # Atmosphere
    RHO_0 = 1.225           # Sea level density (kg/m^3)
    H_SCALE = 8500.0        # Scale height (m)
    
    # Simulation
    DURATION = 6000.0       # Seconds (approx 1 orbit)
    DT = 1.0                # Time step for evaluation

# --- 2. SATELLITE MODEL (The "Spice" is here) ---
class Satellite:
    def __init__(self):
        self.mass = 50.0 # kg
        # Inertia Tensor
        self.I = np.diag([5.0, 5.0, 4.0])
        self.I_inv = np.linalg.inv(self.I)
        
        # Aerodynamics
        self.area = 1.0
        self.cd = 2.2
        
        # SPICE: Center of Pressure Offset (creates torque)
        # Displacement of CoP from CoM in Body Frame
        self.cop_offset = np.array([0.1, 0.05, -0.05]) 

# --- 3. DYNAMICS ENGINE ---
def equations_of_motion(t, state, sat):
    # Unpack State
    r = state[0:3]   # Position (Inertial)
    v = state[3:6]   # Velocity (Inertial)
    q = state[6:10]  # Quaternion (Body -> Inertial) [x, y, z, w]
    w = state[10:13] # Angular Velocity (Body)
    
    # Normalize Quaternion
    q = q / np.linalg.norm(q)
    r_mag = np.linalg.norm(r)
    
    # --- A. GRAVITY (Newton + J2) ---
    # Basic Newton
    accel_grav = -Config.MU * r / (r_mag**3)
    
    # --- B. ATMOSPHERE & DRAG ---
    alt = r_mag - Config.R_EARTH
    if alt > 0:
        rho = Config.RHO_0 * np.exp(-alt / Config.H_SCALE)
    else:
        rho = 0
        
    v_rel = v # Assuming static atmosphere for simplicity
    v_mag = np.linalg.norm(v_rel)
    
    f_drag_inertial = np.zeros(3)
    t_drag_body = np.zeros(3)
    
    if v_mag > 0 and rho > 0:
        # Drag Force
        force_mag = 0.5 * rho * (v_mag**2) * sat.cd * sat.area
        f_drag_inertial = -force_mag * (v_rel / v_mag)
        
        # SPICE: Drag Torque
        # Rotate Force to Body Frame to calculate Cross Product
        rot = R.from_quat(q) # Scipy uses [x,y,z,w]
        f_drag_body = rot.inv().apply(f_drag_inertial)
        t_drag_body = np.cross(sat.cop_offset, f_drag_body)

    # --- C. GRAVITY GRADIENT TORQUE ---
    # T_gg = 3*mu/R^5 * (r_body x I*r_body)
    rot = R.from_quat(q)
    r_body = rot.inv().apply(r)
    t_gg = (3 * Config.MU / (r_mag**5)) * np.cross(r_body, sat.I @ r_body)

    # --- D. CONTROLLER (Nadir Pointing) ---
    # Desired: Z-axis points to Earth Center, Y-axis to Orbit Normal
    z_des = -r / r_mag
    h_vec = np.cross(r, v)
    y_des = -h_vec / np.linalg.norm(h_vec)
    x_des = np.cross(y_des, z_des)
    
    # Target Rotation Matrix
    target_rot_matrix = np.column_stack((x_des, y_des, z_des))
    r_target = R.from_matrix(target_rot_matrix)
    
    # Error Quaternion
    r_current = R.from_quat(q)
    r_error = r_target.inv() * r_current
    q_err = r_error.as_quat()
    
    # Control Law (PD)
    Kp = 0.5
    Kd = 4.0
    # Use vector part of quaternion for error (approx angle)
    # Handle double cover (q = -q)
    if q_err[3] < 0: q_err = -q_err
    t_ctrl = -Kp * q_err[:3] - Kd * w

    # --- E. INTEGRATION ---
    # Linear Acceleration
    accel_total = accel_grav + (f_drag_inertial / sat.mass)
    
    # Angular Acceleration (Euler's Eq)
    # I * w_dot + w x (I * w) = Torques
    t_total = t_gg + t_drag_body + t_ctrl
    w_dot = sat.I_inv @ (t_total - np.cross(w, sat.I @ w))
    
    # Quaternion Derivative
    # q_dot = 0.5 * q * w
    # Create pure quaternion from w
    w_quat = np.array([w[0], w[1], w[2], 0.0])
    # Quaternion multiplication (Hamilton product approximation)
    # q_dot = 0.5 * (Q_current * W_pure)
    # Using simple matrix math for [x,y,z,w]
    qx, qy, qz, qw = q
    wx, wy, wz = w
    q_dot = 0.5 * np.array([
        qw*wx + qy*wz - qz*wy,
        qw*wy - qx*wz + qz*wx,
        qw*wz + qx*wy - qy*wx,
        -qx*wx - qy*wy - qz*wz
    ])
    
    return np.concatenate((accel_total, q_dot, w_dot))

# Wrapper for solve_ivp
def dynamics_wrapper(t, y, sat):
    # solve_ivp passes y as [r, v, q, w]
    # we need to return [dr, dv, dq, dw]
    # My equations_of_motion returned [dv, dq, dw]
    # So we need to reconstruct.
    
    r = y[0:3]
    v = y[3:6]
    
    derivs = equations_of_motion(t, y, sat)
    
    accel = derivs[0:3]
    q_dot = derivs[3:7]
    w_dot = derivs[7:10]
    
    return np.concatenate((v, accel, q_dot, w_dot))

# --- 4. VISUALIZATION (VTK) ---
class Visualizer:
    def __init__(self):
        # Renderer setup
        self.ren = vtk.vtkRenderer()
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.ren_win)
        
        # 1. Earth (Sphere)
        earth_source = vtk.vtkSphereSource()
        earth_source.SetRadius(Config.R_EARTH)
        earth_source.SetThetaResolution(50)
        earth_source.SetPhiResolution(50)
        
        earth_mapper = vtk.vtkPolyDataMapper()
        earth_mapper.SetInputConnection(earth_source.GetOutputPort())
        
        self.earth_actor = vtk.vtkActor()
        self.earth_actor.SetMapper(earth_mapper)
        # Load texture or just color it Blue
        self.earth_actor.GetProperty().SetColor(0.2, 0.4, 0.8) 
        self.earth_actor.GetProperty().SetOpacity(0.8)
        self.ren.AddActor(self.earth_actor)


        # 2. Satellite (STL Model)
        # A. Read the STL file
        reader = vtk.vtkSTLReader()
        reader.SetFileName("sat.stl")
        # Important: Force an update to read file metadata now
        reader.Update() 

        # B. Create a Transform Filter for Scaling/Orientation
        t = vtk.vtkTransform()
        # [IMPORTANT] VISUAL SCALING:
        # The Earth is huge (~6 million meters radius).
        # A real-size satellite STL will be invisible. We must apply a massive fake scale.
        # Start with 10000. If it's too big/small, adjust this number.
        vis_scale = 1000.0 
        t.Scale(vis_scale, vis_scale, vis_scale)
        
        # [OPTIONAL] Fix STL Orientation alignment if needed.
        # If your model comes in facing the wrong way relative to its motion,
        # uncomment and adjust these rotations to align STL axes with Body axes.
        # t.RotateX(90) 
        # t.RotateZ(90)

        # C. Apply transform to the polydata geometry
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(reader.GetOutputPort())
        transformFilter.SetTransform(t)
        transformFilter.Update()

        # D. Mapper and Actor (same as before, but connecting the filter output)
        sat_mapper = vtk.vtkPolyDataMapper()
        # Connect the output of the transform filter, not the raw reader
        sat_mapper.SetInputConnection(transformFilter.GetOutputPort())
        
        self.sat_actor = vtk.vtkActor()
        self.sat_actor.SetMapper(sat_mapper)
        # You might want a metallic color for a real satellite
        self.sat_actor.GetProperty().SetColor(0.8, 0.8, 0.8) # Silver/Greyish
        # Optional beauty settings
        self.sat_actor.GetProperty().SetSpecular(0.5)
        self.sat_actor.GetProperty().SetSpecularPower(20)
        
        self.ren.AddActor(self.sat_actor)
        
        # 3. Trajectory (Line)
        self.points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()
        self.polyData = vtk.vtkPolyData()
        self.polyData.SetPoints(self.points)
        self.polyData.SetLines(self.lines)
        
        trail_mapper = vtk.vtkPolyDataMapper()
        trail_mapper.SetInputData(self.polyData)
        self.trail_actor = vtk.vtkActor()
        self.trail_actor.SetMapper(trail_mapper)
        self.trail_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        self.ren.AddActor(self.trail_actor)
        
        # Camera
        self.ren.SetBackground(0.1, 0.1, 0.1) # Space Black
        
        self.cam = self.ren.GetActiveCamera()
        self.cam.SetPosition(Config.R_EARTH * 3, 0, 0)
        self.cam.SetFocalPoint(0, 0, 0)

    def update_scene(self, r, q):
        # Update Sat Position
        self.sat_actor.SetPosition(r[0], r[1], r[2])
        
        # Update Sat Orientation
        # VTK uses Degrees and [W, X, Y, Z] rotation order usually, 
        # but easier to use a Matrix or Axis-Angle.
        rot = R.from_quat(q)
        axis_angle = rot.as_rotvec()
        angle_rad = np.linalg.norm(axis_angle)
        if angle_rad > 0:
            angle_deg = np.degrees(angle_rad)
            axis = axis_angle / angle_rad
            self.sat_actor.SetOrientation(0,0,0) # Reset
            self.sat_actor.RotateWXYZ(angle_deg, axis[0], axis[1], axis[2])
            
        # Update Trail
        pid = self.points.InsertNextPoint(r)
        if pid > 0:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pid - 1)
            line.GetPointIds().SetId(1, pid)
            self.lines.InsertNextCell(line)
            self.polyData.Modified()
            
        self.cam.SetPosition(r[0]*2, r[1]*2, r[2]*2)
        self.cam.SetFocalPoint(0,0,0)
        self.ren_win.Render()


# --- 5. MAIN ---
def main():
    print("Initializing VTK Simulation...")
    sat = Satellite()
    
    # --- INITIAL CONDITIONS (FIXED) ---
    r0_mag = Config.R_EARTH + 400000.0 # 400km alt
    v_circ = np.sqrt(Config.MU / r0_mag)
    
    # Vectors: Position on X, Velocity on Y -> Counter-Clockwise Orbit
    r0 = np.array([r0_mag, 0, 0])
    v0 = np.array([0, v_circ, 0])
    
    q0 = np.array([0, 0, 0, 1]) # Identity
    w0 = np.array([0.01, 0.01, 0.01]) # Initial tumble
    
    y0 = np.concatenate((r0, v0, q0, w0))
    
    # Pre-Flight Physics Check
    accel_check = Config.MU / (r0_mag**2)
    print(f"Physics Check: Gravity at alt should be ~8.7 m/s^2. Calculated: {accel_check:.2f}")
    
    # Solve Physics beforehand (fast)
    print("Solving Dynamics...")
    t_eval = np.linspace(0, Config.DURATION, 600) # 600 frames
    sol = solve_ivp(dynamics_wrapper, [0, Config.DURATION], y0, t_eval=t_eval, args=(sat,), rtol=1e-6)
    
    print("Starting Animation...")
    vis = Visualizer()
    
    # Animation Loop
    for i in range(len(sol.t)):
        r_step = sol.y[0:3, i]
        q_step = sol.y[6:10, i]
        vis.update_scene(r_step, q_step)
        
        # Simple non-blocking delay
        # In a real app we'd use vtkTimerCallback, but this is simpler for scripts
        # time.sleep(0.01) 
        vis.ren_win.Render()
        vis.iren.ProcessEvents() # Handle window events
        
    print("Done. Press 'q' in window to exit.")
    vis.iren.Start()

if __name__ == "__main__":
    main()