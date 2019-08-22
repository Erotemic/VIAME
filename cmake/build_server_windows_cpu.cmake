set(CTEST_SITE "zeah.kitware.com")
set(CTEST_BUILD_NAME "Windows7_GPU_Master_Nightly")
set(CTEST_SOURCE_DIRECTORY "C:/workspace/VIAME-Seal-CPU")
set(CTEST_BINARY_DIRECTORY "C:/workspace/VIAME-Seal-CPU/build/")
set(CTEST_CMAKE_GENERATOR "Visual Studio 15 2017 Win64")
#set(CTEST_CMAKE_GENERATOR "Visual Studio 15 2017")
#set(CTEST_CMAKE_GENERATOR_PLATFORM "x64")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_PROJECT_NAME VIAME)
set(CTEST_BUILD_MODEL "Nightly")
set(CTEST_NIGHTLY_START_TIME "3:00:00 UTC")
set(CTEST_USE_LAUNCHERS 1)
include(CTestUseLaunchers)
set(OPTIONS 
  "-DCMAKE_BUILD_TYPE=Release"
  "-DPYTHON_EXECUTABLE:PATH=C:/Python36/python.exe"
  "-DPYTHON_INCLUDE_DIR:PATH=C:/Python36/include"
  "-DPYTHON_LIBRARY:PATH=C:/Python36/libs/python36.lib"
  "-DPYTHON_VERSION=3.6"
  "-DVIAME_CREATE_PACKAGE=ON"
  "-DVIAME_ENABLE_CUDNN=OFF"
  "-DVIAME_ENABLE_CUDA=OFF"
  "-DVIAME_ENABLE_CAMTRAWL=OFF"
  "-DVIAME_ENABLE_PYTHON=ON"
  "-DVIAME_ENABLE_GDAL=ON"
  "-DVIAME_ENABLE_SCALLOP_TK=OFF"
  "-DVIAME_ENABLE_PYTORCH=OFF"
  "-DVIAME_ENABLE_PYTORCH-CORE=OFF"
  "-DVIAME_ENABLE_PYTORCH-VISION=OFF"
  "-DVIAME_ENABLE_PYTORCH-MMDET=OFF"
  "-DVIAME_ENABLE_ITK=ON"
  "-DVIAME_ENABLE_TENSORFLOW=ON"
  "-DVIAME_ENABLE_VIVIA=OFF"
  "-DVIAME_ENABLE_SEAL_TK=ON"
  "-DVIAME_KWIVER_BUILD_DIR=C:/tmp/kv4"
  "-DVIAME_PLUGINS_BUILD_DIR=C:/tmp/vm4"
  "-DEXTERNAL_Qt=C:/Qt5/5.12.0/msvc2017_64"
)

set(platform Windows7)
