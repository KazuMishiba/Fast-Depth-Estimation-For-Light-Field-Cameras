# Fast Depth Estimation for Light Field Cameras
Implementation of "Fast Depth Estimation for Light Field Cameras" (IEEE TIP 2020).

This code is based on the code for the paper, and has been modified for public use.
Because of the difference in the libraries used, the estimation results and computation time obtained by using this code are slightly different from those in the paper.

This code is free to use for non-comercial applications.
Please cite the following paper if you use the code for research.

> Kazu Mishiba, "Fast Depth Estimation for Light Field Cameras," IEEE Transactions on Image Processing, vol. 29, pp. 4232-4242, 2020. [Link](https://ieeexplore.ieee.org/document/8985549)

## Environments
The code is tested with
- C++17
- OpenCV 4.5.3
- Boost 1.77.0
- CUDA 10.1
- OpenMP
- Windows 10
- NVIDIA Graphics card with CUDA support
- Visual Studio 2019

## How to build
### Install
1. Install Visual Studio
1. Install OpenCV 
1. Install Boost
1. Install CUDA Toolkit with Visual Studio Integration

We recommend using vcpkg for the installation of OpenCV and Boost.
- .\vcpkg install opencv4:x64-windows
- .\vcpkg install boost:x64-windows
 
### Construct project in Visual Studio
1. Run Visual Studio
1. Select "Create a new project"
1. Select "Empty Project"
1. Add all the code (.cpp, .h, .cu, .cuh) to the project.
1. Set the following Project settings.
1. Build->Build Solution

### Project settings

**Project->Build Customizations**
```
 Check "CUDA XX.X(.targets, .props)"
```
**Project->Properties**
```
 Configuration: All Configurations
 Platform: x64
```
```
Configuration Properties
|-C/C++
| |-Language
| | |-Conformance mode: No
| | |Open MP Support: Yes
| | |C++ Language Standard: ISO C++17 Standard
| |-Precompiled Header
| | |-Precompiled Header: Use
| |-Advanced
|   |-Forced Include File: stdafx.h
|-Linker
  |-Input
    |-Additional Dependencies: cudart_static.lib
```
**stdafx.cpp->properties**
```
Configuration Properties
|-C/C++
  |-Precompiled Header
    |-Precompiled Header: Create
```
**kernel.cu, kernel.cuh->properties**
```
Configuration Properties
|-General
  |-Item Type: CUDA C/C++
```
**Build->Configuration Manager**
```
Active solution configuration: Release
Active solution platform: x64
```

If you installed OpenCV and Boost without vcpkg, 
you have to set 
Additional Include Directories (C/C++->General) and Additional Dependencies (Linker->Input) properly.




## Getting started
1. Build the source code into an executable (FastDepthEstimationForLightFieldCameras.exe).
1. Download light field data from [4D Light Field Dataset](https://lightfield-analysis.uni-konstanz.de/). Here, we use "cotton" data.
1. Place the configuration file `default.json` in the directory containing the EXE file.
1. Set the path to the downloaded data in Path.p1 in `default.json`.
1. Run the EXE file.

**Directory configuration**
```
data
|-training
  |-cotton
    |-input_Cam000.png
    |- ...
    |-input_Cam080.png
    |-parameters.cfg
```

```
code
|-FastDepthEstimationForLightFieldCameras.exe
|-default.json
```

**Set the path in default.json**
```
"Path":
{
   "p1": "path to the data directory",
   "p2": "training",
   "p3": "cotton"
},
```

When setting a path, backslashes should be escaped.
For example, if you want to set 
`C:\Users\Name\data\` to "p1", 
set as `"p1":"C:\\Users\\Name\\data\\"`.


## Using other dataset
The following is an example using a dataset which has $9 \times 9$ viewpoints.

### Viewpoint coordinate
```
|------------------------------------- u
|(-4, -4), ... , (0, -4), ..., (4, -4)
|                  ...
|(-4,  0), ... , (0,  0), ..., (4,  0)
|                  ...
|(-4,  4), ... , (0,  4), ..., (4,  4)
v
```

### Viewpoint index
```
|-------------------------- u
| 000, ... , 004, ..., 008
|            ...
| 036, ... , 040, ..., 044
|            ...
| 072, ... , 076, ..., 080
v
```

### Image name
`baseName + viewpoint index + '.png'`

`baseName` can be set with `Input.baseName` in the configuration file.

If Input.baseName is "img", image names are `img000.png`, ..., `img080.png`.


### Preparation of parameters.cfg
The program reads the following information about the dataset from `parameters.cfg`.
- image_resolution_x_px: Image width
- image_resolution_y_px: Image height
- num_cams_x: Number of viewpoints in the $u$ direction
- num_cams_y: Number of viewpoints in the $v$ direction
- disp_min: Minimum disparity of a scene
- disp_max: Maximum disparity of a scene

It is assumed that num_cams_x and num_cams_y are the same.
We have not checked the behavior when they are different.


4D Light Field Dataset described above has the `parameters.cfg` for all data.
If you want to use other dataset which has no `parameters.cfg` file, create `parameters.cfg` file which contains dataset information described above.

**Example of `parameters.cfg`**
```
[intrinsics]
image_resolution_x_px = 512
image_resolution_y_px = 512

[extrinsics]
num_cams_x = 9
num_cams_y = 9

[meta]
disp_min = -1.2
disp_max = 2.5
```
The minimum and maximum disparities are calculated manually from images or set to reasonable values.

### Preparation of configuration file
Specify the experimental environment in the configuration file `default.json`.

You can use a JSON file with any name and location.
In that case, specify the file path as an argument to the executable.
If it is not specified, the program will search for `default.json` in the directory where the program is located.

If you want to process multiple settings in succession, put multiple JSON files in a directory.
By specifying the directory as the argument of the executable, it processes all the `*.json` files in the directory.


## Configuration file
The configuration file specifies the experimental environment.

In the following, items marked with "(boolean)" must be 0 or 1. 1 means true.
See the paper for the details of Parameter.


### Path (see the following Path setting section)
- p1: String corresponding to #p1.
- p2: String corresponding to #p2.
- p3: String corresponding to #p3.
### Input
- dir: Input directory
- baseName: Base image name.
- estimateAll: If 1, estimate disparities for all viewpoints.
- targetIndex: Index of viewpoint whose disparity is to be estimated. If -1, targetIndex is set to the index of the central viewpoint.
- configFilePath: Path to configuration file.
### Output
- dir: Output directory
- baseNameEstimation: Base save name for estimated disparity.
- baseNameRuntime: Base save name for runtime.
- useViewpointIndexAsSuffix (boolean): If 1, add the index of an estimation target viewpoint followed by the base save name.
- suffixInitialEstimation: Suffix for an initial estimation result to distinguish it from the final result.
- saveResultInitialEstimation (boolean): Save initial estimation disparity.
- saveResultInitialRuntime (boolean): Save runtime for initial estimation disparity.
- saveResultFinalEstimation (boolean): Save final estimation disparity.
- saveResultFinalRuntime (boolean): Save runtime for final estimation (total time of initial estimation and optimization).
- saveAsPFM (boolean): Save disparity to PFM file.
- saveAsPNG (boolean): Save disparity to PNG file (disparity is normalized).
### Parameter
- disparityResolution: $\alpha_{max}$
- gamma: $\gamma$
- lambda: $\lambda$
- sigma: $\sigma$
- W1 (odd number): $W_1$
- W2 (odd number): $W_2$
- t: $t$
- mu0_lowres: $\mu_0$ for downsampled resolution.
- mu0_highres: $\mu_0$ for original resolution.
- kappa: $\kappa$
- tau: $\tau$
- viewSelection:
  - "adaptive": Use view selection written in the paper.
  - "all": Use all viewpoints for estimation. usedViewNum is ignored.
- usedViewNum: Number of viewpoints used for adaptive view selection.
- useOptimization (boolean):
  - 1: The optimization process is performed after the initial estimation process.
  - 0: Only the initial estimation process is performed.
### ReduceData (see the following Data reduction section)
- enable (boolean): Enable data reduction for debugging.
- enableLimitView (boolean): Enable viewpoint limitation.
- useViewLength: Length of a side of viewpoints used.
- enableScaleImageSize (boolean): Enable image scaling.
- scalingRate: Scaling rate.
### Debug
- displayIntermediate (boolean): Display feature map, estimation confidence, etc.
- displayResult (boolean): Display estimated disparity.


## Path setting
Path.p1, p2, and p3 in the configuration file will be replaced in the following settings.

- Input.dir
- Input.baseName
- Input.configFilePath
- Output.dir
- Output.baseNameEstimation
- Output.baseNameRuntime
- Output.suffixInitialEstimation


**Example of path setting**
```
   "Path":
   {
      "p1": "C:\\Users\\Name\\data\\",
      "p2": "training",
      "p3": "dino"
   },
   "Input":
   {
      "dir": "#p1\\#p2\\#p3\\",
      "baseName": "input_Cam",
      "estimateAll": 0,
      "targetIndex": -1,
      "configFilePath": "#p1\\#p2\\#p3\\parameters.cfg"
   },
```
In this example, the input directory becomes `C:\Users\Name\data\training\dino\`.


## Data reduction
For debugging purposes, it is possible to test a smaller number of viewpoints and image sizes than the original data.

If you want to apply data reduction, first set ReduceData.enable to 1, then set ReduceData.enableLimitView to 1 if you want to limit the number of views used, and set ReduceData.enableScaleImageSize to 1 if you want to reduce the image size.

**Example of data reduction**

Suppose the original data is $512 \times 512$ in image size with $9 \times 9$ viewpoints.
```
   "ReduceData":
   {
      "enable" : 1,
      "enableLimitView" : 1,
      "useViewLength" : 5,
      "enableScaleImageSize" : 1,
      "scalingRate" : 0.5
   },
```
In this configuration, the viewpoints used are $5 \times 5$ as follows, and the image size is reduced to $256 \times 256$.

```
|------------------------------------- u
|(-2, -2), ... , (0, -2), ..., (2, -2)
|                  ...
|(-2,  0), ... , (0,  0), ..., (2,  0)
|                  ...
|(-2,  2), ... , (0,  2), ..., (2,  2)
v
```

The image numbers to be loaded are as follows.
```
|-------------------------- u
| 020, ... , 022, ..., 024
|            ...
| 038, ... , 040, ..., 042
|            ...
| 056, ... , 058, ..., 060
v
```

In the program, the indices are reassigned as follows.
```
|-------------------------- u
| 000, ... , 002, ..., 004
|            ...
| 010, ... , 012, ..., 014
|            ...
| 020, ... , 022, ..., 024
v
```

The index assigned to the saved image and the index specified in Input.targetIndex are the indices after reassignment.
For example, the disparity estimation result for the central viewpoint image `input_Cam040.png` is named `result012.png`.
