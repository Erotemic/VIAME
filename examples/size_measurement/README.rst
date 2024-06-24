
===========================
Size Measurement Examples
===========================

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/fish_measurement_example.png
   :scale: 60
   :align: center

Running the Demo
================

This section corresponds to the `size measurement`_ example folder within a VIAME desktop
installation. This folder contains examples covering fish measurement using stereo. This example is currently a work
in progress.

.. _this example online: https://github.com/VIAME/VIAME/tree/master/examples/size_measurement

Run CMake to automatically download the demo data into this example folder.
Alternatively you can download the demo data `directly`_.

.. _directly: https://data.kitware.com/#item/5a8607858d777f068578345e`

Setup:
------

Make sure you build VIAME with `VIAME_ENABLE_PYTHON=True` and
`VIAME_ENABLE_OPENCV=True`.

For simplicity this tutorial will assume that the VIAME source directory is
`[viame-source]` and the build directory is `[viame-build]`. Please modify
these as needeed to match your system setup. We also assume that you have built
VIAME.

Additionally this example requires an extra python dependency to be installed.
On Linux or Windows, 'pip install ubelt'.


Running via the pipeline runner
-------------------------------

To run the process using the sprokit C++ pipeline we use the the pipeline
runner:

::

    # First move to the example directory
    cd [viame-build]/install/examples/size_measurement

    # The below script runs pipeline runner on the measurement_example.pipe
    bash run_measurer.sh


This example runs at about 4.0Hz, and takes 13.3 seconds to complete on a 2017
i7 2.8Ghz Dell laptop.


Running via installed opencv python module 
--------------------------------------------

The above pipeline can alternatively be run as a python script.

::

    # move to your VIAME build directory
    cd [viame-build]
    # Run the setup script to setup the proper paths and environment variables
    source install/setup_viame.sh

    # you may also want to set these environment variables
    # export KWIVER_DEFAULT_LOG_LEVEL=debug
    export KWIVER_DEFAULT_LOG_LEVEL=info
    export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes


You should be able to run the help command

:: 

    python -m viame.processes.opencv.ocv_stereo_demo --help

The script can be run on the demodata via

::

    python -m viame.processes.opencv.ocv_stereo_demo \
        --left=camtrawl_demodata/left --right=camtrawl_demodata/right \
        --cal=camtrawl_demodata/cal.npz \
        --out=out --draw -f


Running via the standalone script
---------------------------------

Alternatively you can run by specifying the path to opencv module (if you
have a python environment you should be able to run this without even building
VIAME)



::

    # First move to the example directory
    cd [viame-source]/examples/size_measurement

    # Run the camtrawl module directly via the path
    python ../../plugins/opencv/python/viame/processes/opencv \
        --left=camtrawl_demodata/left --right=camtrawl_demodata/right \
        --cal=camtrawl_demodata/cal.npz \
        --out=out --draw -f

Without the `--draw` flag the above example, this example runs at about 2.5Hz,
and takes 20 seconds to complete on a 2017 i7 2.8Ghz Dell laptop.

With `--draw` it takes significantly longer (it runs at 0.81 Hz and takes over
a minute to complete), but will output images like the one at the top of this
readme as well as a CSV file.

Note that the KWIVER C++ Sprokit pipline offers a significant speedup (4Hz vs
2.5Hz), although it currently does not have the ability to output the algorithm
visualization.

Calibration File Format
-----------------------

For the npz file format the root object should be a python dict with the
following keys and values:

|
|    R: extrinsic rotation matrix
|    T: extrinsic translation
|    cameraMatrixL: dict of intrinsict parameters for the left camera
|        fc: focal length
|        cc: principle point
|        alpha_c: skew
|    cameraMatrixR: dict of intrinsict parameters for the right camera
|        fc: focal length
|        cc: principle point
|        alpha_c: skew
|    distCoeffsL: distortion coefficients for the left camera
|    distCoeffsR: distortion coefficients for the right camera
|
 
For the mat file, format the root structure should be a dict with the key
`Cal` whose value is a dict with the following items:

|
|    om: extrinsic rotation vector (note rotation matrix is rodrigues(om))
|    T: extrinsic translation
|    fc_left: focal length of the left camera
|    cc_left: principle point
|    alpha_c_left: skew
|    kc_left: distortion coefficients for the left camera
|    fc_right: focal length of the right camera
|    cc_right: principle point
|    alpha_c_right: skew
|    kc_right: distortion coefficients for the right camera
|
