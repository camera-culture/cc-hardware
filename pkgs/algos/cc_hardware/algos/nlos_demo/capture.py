from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHSensor
from cc_hardware.utils.plotting import histogram_gui
from multiprocessing import Process, Value, Array
import sys
import time
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    dataset_dir = "datasets/display-box-13-retroreflector/"
    samples_per_category = 200
    num_categories = 4
    chunk = True
    chunk_size = 100

    sensor = VL53L8CHSensor(debug=True, manual_flush=True)

    trigger = True

    for i in range(num_categories):
        print(f"Category {i}")
        print("\a")
        if trigger:
            input("Press Enter to continue...")

        time.sleep(15)
        print("\a\a")
        

        print(f"Capturing {samples_per_category} samples")
        # time.sleep(5)
        
        if chunk:
            print("Chunking")
            hists = []
            for j in tqdm(range(samples_per_category // chunk_size)):
                print(f"Capturing samples {j*chunk_size} to {(j+1)*chunk_size}")
                chunk_hists = sensor.accumulate(chunk_size, average=False)
                time.sleep(1)
                hists.extend(chunk_hists)
            hists_arr = np.array(hists)
        else:
            print("Not chunking")
            hists = sensor.accumulate(samples_per_category, average=False)
            hists_arr = np.array(hists)

        filepath = dataset_dir + f"histograms_{i}.npy"
        np.save(filepath, hists_arr)

        print(f"Finished capturing {samples_per_category} samples for category {i}")
        
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        print("\a\a")


