import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time  # To measure elapsed time

# Define the function
def f(t):
    # return (np.exp(-2j * np.pi  * t))  # Unit Circle
    # return np.exp(-t) # Original Function
    return np.exp(-t)*(np.exp(-2j * np.pi*1 * t))
    # return  (np.sin(10*t)+np.sin(20*t)+np.sin(30*t))*(np.exp(-2j * np.pi *0.5* t))  # Fourier Transfomr but without integral LOL

# Setup figure
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title(r"Animation of $f(t) = e^{-2 \pi i t}$")  # Raw string for title
ax.set_xlabel("Real")
ax.set_ylabel("Imaginary")

# Initialize plot elements
point, = ax.plot([], [], 'ro', label="Current Point")  # Red point
trail, = ax.plot([], [], 'b-', alpha=0.6, label="Path")  # Blue trail
elapsed_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)  # Text for elapsed time
ax.legend()

# Values for animation
t_values = np.linspace(0, 10, 2500)  # t values from 0 to 1
x_trail, y_trail = [], []

# Start the timer
start_time = time.time()

# Update function
def update(frame):
    t = t_values[frame]
    z = f(t)
    x, y = z.real, z.imag
    x_trail.append(x)
    y_trail.append(y)
    
    # Update the current point and the trail
    point.set_data([x], [y])  # Ensure data is a sequence
    trail.set_data(x_trail, y_trail)
    
    # Update elapsed time
    elapsed_time = time.time() - start_time
    elapsed_text.set_text(f"Elapsed Time: {elapsed_time:.2f} s")
    
    return point, trail, elapsed_text

# Create animation
ani = FuncAnimation(fig, update, frames=len(t_values), interval=1, blit=True)
plt.show()
