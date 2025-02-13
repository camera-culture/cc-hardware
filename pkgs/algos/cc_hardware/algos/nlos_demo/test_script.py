from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHSensor
from cc_hardware.utils.plotting import histogram_gui
from multiprocessing import Process, Value, Array
import sys

# sensor.accumulate(1)
# processes = Array('i', [])
processes = []

def hist_gui_wrapper(num_frames=10000, max_bin=24, ylim=50):
    sensor = VL53L8CHSensor(debug=True)
    histogram_gui(sensor, num_frames=num_frames, max_bin=max_bin, ylim=ylim)
    sensor.close()

def input_wrapper():
    sensor = VL53L8CHSensor(debug=False)
    sys.stdin = open(0)
    while True:
        input_command = input()
        # print(f'command: {input_command}')
        if input_command[0] == 'b':
            sensor.change_num_bins(int(input_command[1:]))
        elif input_command[0] == 'r':
            sensor.change_resolution(int(input_command[1:]))
        # num_bins = int(input_command)
        # global processes
        # print(processes)
        # processes[1].terminate()
        # del processes[1]
        # p2 = Process(target=hist_gui_wrapper, args=(1000, num_bins, 100))
        # p2.start()
        # processes.append(p2)

if __name__ == "__main__":
    # p1 = Process(target=input_wrapper, args=())
    # p1.start()
    # processes.append(p1)
    

    p2 = Process(target=hist_gui_wrapper, args=())
    p2.start()
    processes.append(p2)
    # print(processes)

    # for p in processes:
    #     p.join()

# histogram_gui(sensor, num_frames=1000, max_bin=24, ylim=100)
# sensor.accumulate(1)




