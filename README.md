BEV Demo
========

This demo uses a Hailo-8 device with PETR to process 6 input images from nuScenes dataset.
It annotates these images with 3D bounding boxes and creates Bird's Eye View (BEV) representations.

Pipeline
--------

![Pipeline](./resources/pipeline.png)

Requirements
------------

- hailo_platform==4.18.0
- Pyhailort
- mmdet3d.datasets 


Usage
-----

1. Clone the repository:
    ```shell script
    git clone git clone https://github.com/ronithailo/BEV_Demo.git
            
    cd BEV_Demo
    ```

2. Install dependencies:
    ```shell script
    pip install -r requirements.txt
    ```

3. Download demo resources:
    ```shell script
    ./download_resources.sh
    ```

4. Visualize results:
    To visualize results without running inference, use the following command:
    ```shell script
    ./src/visualize_results.py
    ```

    Arguments:
  
    ``-f, --file``: scene data file path.

    For more information:
    ```shell script
    ./src/visualize_results.py -h
    ```
    Example 
    **Command**
    ```shell script
    ./src/visualize_results.py -f resources/results/scenes_data.json
    ```

Run Inference
-------------

1. Data creation:
    - Download the full dataset from https://www.nuscenes.org/nuscenes#download

2. Cloning PETR repository:
    ```shell script
    git clone https://github.com/megvii-research/PETR.git && cd PETR && git checkout f7525f9
    ```

3. Pickle creation:
    - To generate a pickle file for sweep data using the generate_sweep_pkl.py script from the PETR GitHub repository - 
    https://github.com/megvii-research/PETR/blob/main/tools/generate_sweep_pkl.py

4. Data preparation:
    To prepare your data using the data_preparation script:
    - Update Paths: Update all paths in the data_preparation.py script file to match your environment.

        ```shell script
        ./src/data_preparation.py -p <path to the PETR folder> -c <path to config_file.py>
        ```
    - Upon completion of the script, you will find .pt files generated as a result of the data preparation process.

5. Run inference: 
    ```shell script
    ./src/bev.py -m <model_path> -i <input_path> -d <data_path> -f <wanted_FPS> --infinite-loop -n <number_of_scenes>
    ```
    For optimal visibility of 3D boxes, aim for an FPS range of 1 to 4. Higher FPS values might hinder clear observation, given that nuScenes samples were captured at 2 FPS. Higher FPS settings can create an illusion of faster vehicle movement.

Arguments
---------
- ``-f, --fps``: Wanted FPS (1 - 25). 
- ``--infinite-loop``: Run the demo in infinite loop.
- ``-i, --input``: path to the input folder, where all the .pt files are.
- ``-m, --models``: path to the models folder.
- ``-d, --data``: path to the data folder, where the nuSence dataset is.
- ``-n, --number-of-scenes``: number of scenes to run.

For more information:
```shell script
./src/bev.py -h
```
Example 
-------
**Command**
```shell script
./src/bev.py --f 5 -n 2
```

Additional Notes
----------------
- The demo was only tested with ``HailoRT v4.18.0``
- Ran the demo on: Dell PC (Model: Latitude 5431), with CPU (Model: 12th Gen Intel(R) Core(TM) i7-1270P).

Disclaimer
----------
This code demo is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code demo. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code demo or any part of it. If an error occurs when running this demo, please open a ticket in the "Issues" tab.

This demo was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The demo might work for other versions, other environment or other HEF file, but there is no guarantee that it will.