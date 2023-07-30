# -----------------------------------------------------------------------------
# 
# This script plots functions describing the trajectories of freely falling
# atoms in an atom interferometer.
#
# Author: Max Melching
# Source: https://github.com/MaxMelching/physics_plots
# 
# -----------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn-deep')  # Personal preference



def upper(t_pulse: float,
          t_wait: float = 0,
          z0: float = 0,
          v0: float = 0,
          k_eff: float = 0,
          g: float = 9.81,
          num: int = 100
          ) -> tuple:
    """
    Compute upper trajectory.
    """

    times = np.linspace(0, 2 * t_wait + 2 * t_pulse, num)
    results = np.zeros(times.size)

    # Create masks to split times between pulses
    mask1 = times <= t_wait  # Before first Pi/2
    mask2 = (times > t_wait) & (times <= t_wait + t_pulse)  # Pi/2 - Pi
    mask3 = (times > t_wait + t_pulse)\
            & (times <= t_wait + 2 * t_pulse) # Pi - Pi/2
    mask4 = times > t_wait + 2 * t_pulse  # After second Pi/2

    # Compute trajectories for each time interval
    results[mask1] = z0 + v0 * times[mask1] - g / 2 * times[mask1] ** 2
    results[mask2] = z0 + v0 * times[mask2] - g / 2 * times[mask2] ** 2\
                     + k_eff * (times[mask2] - t_wait)
    results[mask3] = z0 + v0 * times[mask3] - g / 2 * times[mask3] ** 2\
                     + k_eff * t_pulse
    results[mask4] = z0 + v0 * times[mask4] - g / 2 * times[mask4] ** 2\
                     + k_eff * t_pulse

    return times, results



def lower(t_pulse: float,
          t_wait: float = 0,
          z0: float = 0,
          v0: float = 0,
          k_eff: float = 0,
          g: float = 9.81,
          num: int = 100
          ) -> tuple:
    """
    Compute lower trajectory.
    """

    times = np.linspace(0, 2 * t_wait + 2 * t_pulse, num)
    results = np.zeros(times.size)

    # Create masks to split times between pulses
    mask1 = times <= t_wait  # Before first Pi/2
    mask2 = (times > t_wait) & (times <= t_wait + t_pulse)  # Pi/2 - Pi
    mask3 = (times > t_wait + t_pulse)\
            & (times <= t_wait + 2 * t_pulse) # Pi - Pi/2
    mask4 = times > t_wait + 2 * t_pulse  # After second Pi/2

    # Compute trajectories for each time interval
    results[mask1] = z0 + v0 * times[mask1] - g / 2 * times[mask1] ** 2
    results[mask2] = z0 + v0 * times[mask2] - g / 2 * times[mask2] ** 2
    results[mask3] = z0 + v0 * times[mask3] - g / 2 * times[mask3] ** 2\
                     + k_eff * (times[mask3] - t_wait - t_pulse)
    results[mask4] = z0 + v0 * times[mask4] - g / 2 * times[mask4] ** 2\
                     + k_eff * t_pulse

    return times, results



if __name__ == '__main__':
    pulse_time = 1  # Times when pulses occur
    wait_time = pulse_time / 4  # Waiting time before, after pulses
    v_init = -1  # Initial momentum p_init (we set m = 1)
    v_added = -24  # = k_eff, momentum added by pulses.
                   # More separation for larger values.
    G = 9.81  # Gravitational constant


    times, traject = upper(pulse_time, wait_time, v0=v_init, k_eff=v_added)
    plt.plot(times, traject)

    times, traject = lower(pulse_time, wait_time, v0=v_init, k_eff=v_added)
    plt.plot(times, traject)

    # Plot dashed lines indicating laser pulses
    plt.plot(2 * [wait_time], [0, traject[-1]], 'r--')
    plt.plot(2 * [wait_time + pulse_time], [0, traject[-1]], 'r--')
    plt.plot(2 * [wait_time + 2 * pulse_time], [0, traject[-1]], 'r--')

    plt.xlabel('$t$')
    plt.ylabel('$z$')
    plt.title('Atom Interferometer Trajectories')

#     plt.savefig('pictures/ai_trajec.png', bbox_inches='tight')

    plt.show()