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