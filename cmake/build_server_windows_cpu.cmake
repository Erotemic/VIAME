set(CTEST_SITE "zeah.kitware.com")
set(CTEST_BUILD_NAME "Windows7_GPU_Master_Nightly")
set(CTEST_SOURCE_DIRECTORY "C:/workspace/VIAME-Windows-CPU-Release")
set(CTEST_BINARY_DIRECTORY "C:/workspace/VIAME-Windows-CPU-Release/build/")
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
  "-DVIAME_CREATE_PACKAGE=ON"
  "-DVIAME_ENABLE_CUDNN=OFF"
  "-DVIAME_ENABLE_CUDA=OFF"
  "-DVIAME_ENABLE_CAMTRAWL=ON"
  "-DVIAME_ENABLE_DIVE=ON"
  "-DVIAME_ENABLE_PYTHON=ON"
  "-DVIAME_ENABLE_PYTHON-INTERNAL=ON"
  "-DVIAME_ENABLE_GDAL=ON"
  "-DVIAME_ENABLE_SCALLOP_TK=OFF"
  "-DVIAME_ENABLE_PYTORCH=ON"
  "-DVIAME_PYTORCH_VERSION=1.9.1"
  "-DVIAME_ENABLE_PYTORCH-INTERNAL=OFF"
  "-DVIAME_ENABLE_PYTORCH-VIS-INTERNAL=OFF"
  "-DVIAME_ENABLE_PYTORCH-MMDET=OFF"
  "-DVIAME_ENABLE_PYTORCH-NETHARN=OFF"
  "-DVIAME_ENABLE_PYTORCH-PYSOT=OFF"
  "-DVIAME_KWIVER_BUILD_DIR=C:/tmp/kv2"
  "-DVIAME_PLUGINS_BUILD_DIR=C:/tmp/vm2"
)

set(platform Windows7)
