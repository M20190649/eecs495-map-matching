# EECS 495 Geospatial Vision
## Nick Paras | Kapil Garg
## Probe Data Analysis for Road Slope

### Install python 3.6 and conda
We use `Python 3.6` with the `conda` package manager. To install this, we recommend following [this guide](https://www.solarianprogrammer.com/2016/11/29/install-opencv-3-with-python-3-on-macos/) up until you must run `conda info`.

### Create a conda env with required packages
Open Terminal and navigate to the project folder. Run `conda create --name probe-data -f environment.yml` to create a new conda env and install all the required packages for our project. Then, run `source activate probe-data` to start the virtual environment. Note that you will need to source the environment each time you wish to use or view our code.

### Working with Jupyter Notebooks
We have provided two notebooks to document our thought process:

1. map_matching.ipynb: exploratory data analysis and implementation/visualization of point-to-link matching
2. trajectory_map_matching.ipynb: implementation/visualization of point-to-link with heading matching
3. st_map_matching.ipynb: implementation/visualization of st-matching found in [Map-Matching for Low_sampling-Rate GPS Trajectories](https://www.microsoft.com/en-us/research/publication/map-matching-for-low-sampling-rate-gps-trajectories/?from=http%3A%2F%2Fresearch.microsoft.com%2Fapps%2Fpubs%2Fdefault.aspx%3Fid%3D105051)

To launch a notebook, run `jupyter notebook notebookname.ipynb` where `notebookname.ipynb` is replaced with one of the above names. This will start a Jupyter notebook server and launch the notebook in your defauly web browser.
### Running executable script
We have additionally included executable scripts for each of our notebooks about. Respectively, they are:

1. point_to_link.py
2. trajectory_map_matching.py
3. st_map_matching.py

Run them with the following: `python script_name.py` where `script_name.py` is the name of the script from above.