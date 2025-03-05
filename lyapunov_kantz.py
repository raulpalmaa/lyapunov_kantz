####################################################################
####################################################################
# Refs.
# 1 : A robust method to estimate the maximal Lyapunov exponent of a time series - https://doi.org/10.1016/0375-9601(94)90991-1
# 2 : Nonlinear time series analysis - H. Kantz & T. Schreiber- Pg. 70
####################################################################
####################################################################

import time
import concurrent.futures
import numpy as np

# Start the timer to measure execution time
start = time.perf_counter()


# Henon map function to generate the next state in the system (2D map)
def henon(xy, a=1.4, b=0.3):
    """
    Henon map function. Returns the next (x, y) state based on the given equations:
    x_{n+1} = 1 - a * x_n^2 + y_n
    y_{n+1} = b * x_n
    """
    x, y = xy
    xit = 1 - a * x**2 + y  # x equation
    yit = b * x  # y equation
    return xit, yit


# Function to find neighborhoods within a distance threshold and compute their distances
def makelist(lxy, tau):
    """
    Creates a list of distances between points in the time series lxy that are within
    a distance threshold (eps) from each other. It uses a time-delay embedding of tau steps.
    """
    eps = 0.008  # Distance threshold for finding similar points
    lista = [[] for _ in range(len(lxy))]  # Initialize empty lists for each point

    # Compare all pairs of points and check if their distance is below the threshold
    for K in range(len(lxy) - tau):
        for i in range(len(lxy) - tau):
            if K != i:  # Avoid comparing the same point with itself
                # Calculate the Euclidean distance between the two points
                distance = np.linalg.norm(lxy[i, :] - lxy[K, :])
                if distance <= eps:  # If within the threshold, consider it a similar neighborhood
                    # Add the distance of future points (with time delay tau)
                    lista[K].append(np.linalg.norm(lxy[i + tau, :] - lxy[K + tau, :]))
    
    # Remove empty neighborhoods (no similar points) from the list
    lista = [ll for ll in lista if ll != []]
    return lista


# Generate the time series using the Henon map
trs = 12000  # Length of transient
t = int(trs / 4)  # Length of the actual time series

# Initialize the first time series `xp`
xp = np.zeros((trs + 1, 2))  # Array to store the 2D time series
xp[0] = 0.353 * np.ones(2)  # Initial state of the series

# Generate the time series `xp` using the Henon map
for i in range(trs):
    xp[i + 1] = henon(xp[i])

# Initialize the second time series `xyn` starting from the last point of `xp`
xyn = np.zeros((t + 1, 2))
xyn[0] = xp[-1]  # Start from the last point of the first series

# Generate the second time series `xyn` using the Henon map
for i in range(t):
    xyn[i + 1] = henon(xyn[i])


# Function to compute the Lyapunov exponent using the neighborhoods of points
def lyaS(entries):
    """
    Computes S for a given value of tau using the neighborhoods
    and distances calculated from the time series. Eq. 2.3 of (1).
    """
    tau = entries[0]  # Time delay for the neighborhood calculation
    xy_tau = np.zeros((len(xyn) - tau, 2))  # Create a new time series with time delay tau
    xy_tau[:, 0] = xyn[tau:, 0]  # Delay the first dimension (x)
    xy_tau[:, 1] = xyn[:len(xy_tau), 0]  # Use the first dimension (x) for the second dimension (y)
    
    # Get neighborhoods for points in the time series
    diffs = makelist(xy_tau, tau)[:500]  # Take the first 500 neighborhoods (can be adjusted)
    
    # Initialize sum for the Lyapunov exponent calculation
    S = 0

    # Loop through the neighborhoods and compute the logarithmic growth rate
    for neighborhood in diffs:
        soma = 0  # Sum of the distances
        Us = 0    # Number of similar points in the neighborhood
        for k in neighborhood:
            soma += k  # Add the distance for each point in the neighborhood
            Us += 1    # Increment the number of similar points
        
        # Calculate the logarithmic average growth rate for this neighborhood
        S += np.log(soma / Us)

    # Normalize by the number of neighborhoods
    S /= len(diffs)
    return S


# List of tau values for which we want to compute the Lyapunov exponent
taus = [1, 5, 10]

# Parallelize the computation of Lyapunov exponents for the different tau values
rss = []  # List to store the Lyapunov exponents for each tau
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Prepare the tasks for the executor (one task for each tau value)
    yns = [[i] for i in taus]
    results = executor.map(lyaS, yns)  # Compute the Lyapunov exponent for each tau in parallel
    
    # Collect the results from each task
    for result in results:
        rss.append(result)

# Convert the results to a numpy array and save them to a file
S = np.array(rss)
np.savetxt("lyapunov_kant.dat", S)

# Measure the execution time and display the results
finish = time.perf_counter()
print('Approx. Lyapunov exponent is: ', np.polyfit(taus, S, 1)[0])  # Estimate the Lyapunov exponent from the slope (see Fig. 2 of (1))
print(f'Finished in {round(finish - start, 2)} second(s)')
