# k-distance graph

The analysis is intended to assist the user in determining the parameter "epsilon" for DBSCAN analysis.</br>

- Calculate k nearest neighbors
- Display them as k-distance graphs
- Calculate knee-point with kneed[1] &rarr; get epsilon
- Before knee-point calculation the curve is low-pass filtered and normalized

**Requirements** (installed in Anaconda shell): numpy, h5py, tqdm, matplotlib, kneed</br>
**Input file:** Picasso[2] localization hdf5</br>
**Execution:** python k_distance_graph.py k_distance_graph_config.ini</br>

**Config file:**
INPUT_FILES</br>
path_A - define the paths to the picasso localization hdf5 files</br>
PARAMETERS</br>
pixel_size - pixel size of camera in nm</br>
camera_size - number of pixels in a row on the camera chip (e.g.256)</br>
max_k - maximum k-nearest neighbor to be calculated (e.g. k = 3 -> k = 1, 2, 3 are calculated)</br>
calc_knee_point - if knee-point should be detected ("yes" / "no)</br>
low_pass_width - low-pass filtering of k-distance graph (1 = no filtering)</br>
save_plots - if k-distance graphs should be saved as images ("yes" / "no")</br>
SAVE_DIR</br>
save_dir - directory to save results</br>

**Links:**</br>
[1] https://github.com/arvkevi/kneed</br>
[2] https://github.com/jungmannlab/picasso</br>
