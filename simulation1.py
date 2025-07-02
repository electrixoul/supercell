import math

# Physical parameters (SI units)
mass = 13600               # kg  (≈30,000 lb)
length = 6.2               # m   (≈20.5 ft)
diameter = 0.8             # m   (≈31.5 in)
radius = diameter / 2

# Fluid properties of fresh water at ~20 °C
rho_water = 1000           # kg/m³  Density
mu_water = 0.001           # Pa·s   Dynamic viscosity (not used in quadratic model)

# Hydrodynamic drag parameters
C_d = 0.82                 # Drag coefficient for an axially‑moving blunt cylinder
                           # (Tandfonline DOI:10.1080/00223131.2022.2064357)

# Derived geometry
A_front = math.pi * radius**2               # Frontal area (m²)
V_cyl   = A_front * length                  # Volume (m³)

# Constant forces
g = 9.81
buoyancy = rho_water * V_cyl * g
weight   = mass * g
Fg_minus_b  = weight - buoyancy             # Net downward weight in water (N)

# Convenience coefficients for analytic integral
k1 = 0.5 * rho_water * C_d * A_front / mass # (1/m)
b  = Fg_minus_b / mass                      # (m/s²)

# Initial and final speeds (m/s)
v0 = 450
v1 = 100

# Closed‑form depth from ∫ v/(b – k₁v²) dv
depth = (1 / (2 * k1)) * math.log( (b - k1 * v0**2) / (b - k1 * v1**2) )

print(f"Required water depth ≈ {depth:0.1f} m")
