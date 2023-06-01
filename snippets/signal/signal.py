def rolling_window(lst, window_size):
    """Generate a rolling window of a given size over a list.

    Args:
        lst (list): Input list
        window_size (int): Size of the rolling window

    Returns:
        list: List of sublists, each containing a rolling window of size `window_size` over the input list
    """
    result = []
    for i in range(len(lst) - window_size + 1):
        result.append(lst[i:i+window_size])
    return result

if __name__ == '__main__':
    signal = [1,2,1,2,1,2,2,2,1,1,2,1,1,2,1,50,48,49,50,50,50,51,51,1,2,3,1,2,2,1,2,1,2,1]

    # Detect a change in the signal using the derivative to detect significant change
    diff = [signal[i+1] - signal[i] for i in range(len(signal)-1)]

    wait_time = 0 # Initialize auxiliary variable. `wait_time` is used to make sure that a spike is not detecte multiple times
    # To remove noise in the signal, sum the derivatives using a rolling window
    for i in rolling_window(diff, 5):
         wait_time += 1
         if sum(i) > 3 and wait_time > 5:
              print("a spike has happened")
              wait_time = 0






def freq_analysis(time_data, fs):
    """Calculates using FFT, the values to plot in the frequency domain.

    Args:
        time_data (list): List with he time series data
        fs (float): Sampling frequency in Hz=samples/second. fs=1/Ts. Ts = sampling rate, how many seconds per sample. 

    Returns:
        positive_freqs (np.ndarray): X axis for the plot
        positive_yf (np.ndarray): Y axis for the plot
    """
    yf = fft(time_data)
    freqs = fftfreq(len(time_data), 1/fs)

    # Only return positive freqs
    positive_freqs = freqs[np.where(freqs >= 0)]
    positive_yf = yf[np.where(freqs >= 0)]

    return positive_freqs, positive_yf

def plot_timeseries_freq(xts, yts, xf, yf):
    """Generates a plot showing both the time domain and the frequency domain.

    Args:
        xts (list): Time domain X axis
        yts (list): Time domain Y axis
        xf (np.ndarray): Frequency domain X axis
        yf (np.ndarray): Frequency domain Y axis
    """
    # ax1 for time domain, ax2 for frequency domain
    fig, (ax1, ax2) = plt.subplots(2,1, constrained_layout=True)

    # Time domain in colour red
    ax1.plot(xts, yts, color="red")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Frequency domain in colour blue (default color)
    ax2.plot(freqs, np.abs(yf))
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude')
    ax2.axis(xmin=0, xmax=0.0001, ymin=0, ymax=0.2*1e8) # A zoom is defined to show relevant frequencies.

    plt.show()