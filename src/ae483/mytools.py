"""Functions that you write.

Add your own code here, not in tools.py. Keeping them separate means you can
`git pull` our updates all semester without conflicts.

Test code
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Makes everything in tools.py available here, just as if your code were at the
# bottom of that file.
from ae483.tools import *


def transform_data_mocap(raw_data):
    # Copy raw data
    data = {}
    for key, val in raw_data.items():
        data[key] = val.copy()

    # Define parameters
    d_1 = 0.016 
    d_2 = 0.009 
    
    # Pose of drone body frame in active marker frame
    R_inA_ofB = np.eye(3)                   # <-- fixed?
    p_inA_ofB = np.array([0 , 0 , -d_1])      # <-- fixed?

    ####################################
    # START OF ANALYSIS AT TIME STEP 0
    #
    
    # Pose of drone world frame in active marker frame
    R_inA_ofW = np.eye(3)                   # <-- fixed?
    p_inA_ofW = np.array([0 , 0, -d_2 - d_1])      # <-- Fixed?

    # Get measurements of (x, y, z) and (psi, theta, phi) from mocap
    x, y, z = data['x'][0], data['y'][0], data['z'][0]
    psi, theta, phi = data['yaw'][0], data['pitch'][0], data['roll'][0]

    # Convert yaw, pitch, roll to rotation
    rotation = Rotation.from_euler('ZYX', [psi, theta, phi])

    # Pose of active marker frame in mocap world frame
    R_inQ_ofA = rotation.as_matrix()                  # <-- fixed?
    p_inQ_ofA = np.array([x, y, z])      # <-- fixed?

    # Pose of drone world frame in mocap world frame
    R_inQ_ofW = R_inQ_ofA @ R_inA_ofW                   # <-- fixed?
    p_inQ_ofW = p_inQ_ofA + R_inQ_ofA @ p_inA_ofW      # <-- fixed?
    
    # Pose of mocap world frame in drone world frame
    R_inW_ofQ = R_inQ_ofW.T                # <-- fixed?
    p_inW_ofQ = -R_inW_ofQ @ p_inQ_ofW      # <-- fixed?

    #
    # END OF ANALYSIS AT TIME STEP 0
    ####################################

    for i in range(len(data['time'])):

        ####################################
        # START OF ANALYSIS AT TIME STEP i
        #

        # Get measurements of (x, y, z) and (psi, theta, phi) from mocap
        x, y, z = data['x'][i], data['y'][i], data['z'][i]
        psi, theta, phi = data['yaw'][i], data['pitch'][i], data['roll'][i]

        # Convert yaw, pitch, roll to rotation
        rotation = Rotation.from_euler('ZYX', [psi, theta, phi])

        # Pose of active marker deck in mocap world frame
        R_inQ_ofA = rotation.as_matrix()                  # <-- fixed?
        p_inQ_ofA = np.array([x, y, z])  # <-- fixed?

        # Pose of drone body frame in mocap world frame
        R_inQ_ofB = R_inQ_ofA @ R_inA_ofB                   # <-- fixed?
        p_inQ_ofB = p_inQ_ofA + R_inQ_ofA @ p_inA_ofB      # <-- fixed?

        # Pose of drone body frame in drone world frame
        R_inW_ofB = R_inW_ofQ @ R_inQ_ofB                   # <-- fixed?
        p_inW_ofB = p_inW_ofQ + R_inW_ofQ @ p_inQ_ofB      # <-- fixed?

        # Replace measurements of (x, y, z) and (phi, theta, psi) from mocap
        data['x'][i], data['y'][i], data['z'][i] = p_inW_ofB[0], p_inW_ofB[1], p_inW_ofB[2]                  # <-- fixed?
        data['yaw'][i], data['pitch'][i], data['roll'][i] = Rotation.from_matrix(R_inW_ofB).as_euler('ZYX')      # <-- fixed?

        #
        # END OF ANALYSIS AT TIME STEP i
        ####################################
    
    # Return the result
    return data
    raise NotImplementedError('See Lab 2, step 2.5.')


def precise_time_shift(raw_data_mocap, t, z_drone, t_shift_guess,
                       do_transform=True, t_min_offset=0.,
                       t_shift_rng=1., t_shift_res=0.05):
    """
    Find a value of t_shift that precisely aligns mocap data with
    drone data by sampling candidate values in the interval

        [t_shift_guess - t_shift_rng, t_shift_guess + t_shift_rng]
    
    at the resolution t_shift_res and returning the value that
    produces the minimum RMSE between z_drone and z_mocap.
    """
    
    # Create an array of time shifts over which to searchs
    t_shifts = np.linspace(
    t_shift_guess - t_shift_rng,                        # lower bound
    t_shift_guess + t_shift_rng,                        # upper bound
    int(1 + np.ceil((2 * t_shift_rng) / t_shift_res)),  # number of samples
    )

    # Print a warning if computation time is likely to be long
    if len(t_shifts) > 50:
        print(f'========================================================================')
        print(f'WARNING (sync_data_mocap)\n')
        print(f'Searching the interval\n')
        print(f'  t_shift_bnds=[{min(t_shifts):4.2f}, {max(t_shifts):4.2f}]\n')
        print(f'with the resolution\n')
        print(f'  t_shift_res={t_shift_res:4.2f}\n')
        print(f'means computing RMSE for {len(t_shifts)} possible time shifts, which is likely')
        print(f'to be slow. You may want to decrease the size of the search interval')
        print(f'or increase the resolution.')
        print(f'========================================================================\n')

    # Create an array to hold the RMSE for each time shift
    RMSEs = np.empty_like(t_shifts)

    # Find the RMSE for each time shift
    for i, t_shift in enumerate(t_shifts):
        # Resample mocap data with time shift
        resampled_data_mocap = resample_data_mocap(
            raw_data_mocap,
            t,
            t_shift=t_shift,
            t_min_offset=t_min_offset,
        )

        # Transform mocap data
        if do_transform:
            transformed_data_mocap = transform_data_mocap(resampled_data_mocap)
        else:
            transformed_data_mocap = resampled_data_mocap

        # Get z estimate from mocap data
        z_mocap = transformed_data_mocap['z']

        # Find RMSE between z_mocap and z_drone
        RMSEs[i] = np.sqrt(np.mean((z_mocap - z_drone)**2))

    # Find the index of the minimum RMSE
    i_min = np.argmin(RMSEs)    

    # Find the minimum RMSE
    RMSE_min = min(RMSEs)      

    # Find the time shift that gives the minimum RMSE
    t_shift_min = t_shifts[i_min]   

    # Store result for later comparison
    t_shift_min_stored = t_shift_min
        
    # Return the result
    return t_shift_min


def sync_data_mocap(raw_data_mocap, t, z_drone,
                    do_transform=True, t_min_offset=0.,
                    t_shift_rng=1., t_shift_res=0.05):

    # Get rough guess at time shift
    t_shift_guess = rough_time_shift(
        raw_data_mocap,
        t,
        z_drone,
        t_min_offset=t_min_offset,
    )

    # Refine the rough guess to get the time shift that
    # minimizes RMSE between z_drone and z_mocap
    t_shift_min = precise_time_shift(
        raw_data_mocap,
        t,
        z_drone,
        t_shift_guess=t_shift_guess,
        do_transform=do_transform,
        t_min_offset=t_min_offset,
        t_shift_rng=t_shift_rng,
        t_shift_res=t_shift_res,
    )

    # Resample mocap data with this time shift
    resampled_data_mocap = resample_data_mocap(
        raw_data_mocap,
        t,
        t_shift=t_shift_min,
        t_min_offset=t_min_offset,
    )
    
    # Transform mocap data
    if do_transform:
        transformed_data_mocap = transform_data_mocap(resampled_data_mocap)
    else:
        transformed_data_mocap = resampled_data_mocap

    # Return the result
    return transformed_data_mocap