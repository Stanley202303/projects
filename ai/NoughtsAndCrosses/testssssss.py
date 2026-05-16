import numpy as np
import matplotlib.pyplot as plt

# Parameters
m_lego = 0.18  # kg
r_wheel = 0.024  # m
mu = 3  # high-µ tires
g = 9.81  # m/s²

# Motor parameters
V_motor = 11.1  # V
I_motor = 3  # A per motor, 2 motors
P_motor = V_motor * I_motor * 2  # total power in Watts

# Abarth parameters
m_abarth = 1000  # kg
P_abarth = 134000  # W

# Time array for plotting (up to 3 seconds for 0-60 km/h)
t = np.linspace(0, 3, 500)

# LEGO car acceleration: combine torque-limited at very low speed, then traction-limited
v_lego = np.zeros_like(t)
v_max = 16.67  # 60 km/h in m/s
a_traction = mu * g

for i in range(1, len(t)):
    v_prev = v_lego[i-1]
    # Power-limited acceleration (avoid division by zero)
    a_power = P_motor / (m_lego * max(v_prev, 0.01))
    # Limit by traction
    a = min(a_power, a_traction)
    v_lego[i] = v_prev + a * (t[i] - t[i-1])
    if v_lego[i] >= v_max:
        v_lego[i:] = v_max
        break

# Abarth acceleration: power-limited
v_abarth = np.sqrt(2 * P_abarth / m_abarth * t)
v_abarth = np.minimum(v_abarth, v_max)

# Convert to km/h
v_lego_kmh = v_lego * 3.6
v_abarth_kmh = v_abarth * 3.6

# Plot
plt.figure(figsize=(10,6))
plt.plot(t, v_lego_kmh, label='LEGO Car (high-µ tires, modded)', linewidth=2, color='red')
plt.plot(t, v_abarth_kmh, label='Abarth 695', linewidth=2, color='blue')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Speed (km/h)', fontsize=12)
plt.title('0-60 km/h Acceleration Curves: LEGO Car vs Abarth 695', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()
