"""
Plot data coming from a stream
"""
import matplotlib.pyplot as plt

# Initialize empty list for data
data = []

# Set the maximum number of data points to store
max_points = 50

for i in range(10000):
    data.append(i)
    
    # If the number of data points exceeds the maximum, remove the oldest data
    if len(data) > max_points:
        data.pop(0)

    # Clear the plot
    plt.clf()

    # Plot the data
    plt.plot(data)

    # Show the plot
    plt.show(block=False)

    # Pause for a short amount of time
    plt.pause(0.01)
    
