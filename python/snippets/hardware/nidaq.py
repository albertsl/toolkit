"""
Example code to connect to National Instrument's Analogic to Digital Converter
National Instrument's NI-DAQ mx software is required to connect to the machine.
"""
import nidaqmx

if __name__ == "__main__":
    # Innitialize connection to the device
    task = nidaqmx.Task()

    # Add analog input channel(s) to the task
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0")

    s_rate = "5 seconds"
    dict_sample_rates = {"5 seconds": 1/5, "10 seconds": 1/10, "30 seconds": 1/30, "1 minute": 1/60, "5 minutes": 1/(60*5), "10 minutes": 1/(60*10), "15 minutes": 1/(60*15)}
    # Configure the task with the desired sampling clock source. rate is in samples per second
    task.timing.cfg_samp_clk_timing(rate=dict_sample_rates[s_rate], source='OnboardClock')
    
    for i in range(10):
        data = task.read()