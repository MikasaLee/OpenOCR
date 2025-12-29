import os


data_dir = r'/a800data1/lirunrui/origin_datasets/bchw_dataset/bchw_train'
for dirpath, dirnames, filenames in os.walk(data_dir + '/', followlinks=True):
    print(f"dirpath: {dirpath}")
    print(f"dirnames: {dirnames}")
    print(f"filenames: {filenames}")
    print("=" * 50)
