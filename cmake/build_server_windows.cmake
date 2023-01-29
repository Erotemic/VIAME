set(CTEST_SITE "zeah.kitware.com")
set(CTEST_BUILD_NAME "Windows7_GPU_Master_Nightly")
set(CTEST_SOURCE_DIRECTORY "C:/workspace/VIAME-Windows-GPU-Release")
set(CTEST_BINARY_DIRECTORY "C:/workspace/VIAME-Windows-GPU-Release/build/")
set(CTEST_CMAKE_GENERATOR "Visual Studio 16 2019")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_PROJECT_NAME VIAME)
set(CTEST_BUILD_MODEL "Nightly")
set(CTEST_NIGHTLY_START_TIME "3:00:00 UTC")
set(CTEST_USE_LAUNCHERS 1)
include(CTestUseLaunchers)
set(OPTIONS 
  "-DCMAKE_BUILD_TYPE=Release"
  "-DCUDA_NVCC_EXECUTABLE:PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/bin/nvcc.exe"
  "-DCUDNN_ROOT_DIR:PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3"
  "-DVIAME_FIXUP_BUNDLE=ON"
  "-DVIAME_ENABLE_CUDNN=ON"
  "-DVIAME_ENABLE_CUDA=ON"
  "-DVIAME_ENABLE_DIVE=ON"
  "-DVIAME_ENABLE_FFMPEG=ON"
  "-DVIAME_ENABLE_VIVIA=ON"
  "-DVIAME_ENABLE_PYTHON=ON"
  "-DVIAME_ENABLE_PYTHON-INTERNAL=ON"
  "-DVIAME_PYTHON_VERSION=3.6.15"
  "-DVIAME_ENABLE_GDAL=ON"
  "-DVIAME_ENABLE_SCALLOP_TK=OFF"
  "-DVIAME_ENABLE_PYTORCH=ON"
  "-DVIAME_ENABLE_PYTORCH-INTERNAL=OFF"
  "-DVIAME_PYTORCH_VERSION=1.10.2"
  "-DVIAME_ENABLE_PYTORCH-VIS-INTERNAL=ON"
  "-DVIAME_ENABLE_PYTORCH-MMDET=ON"
  "-DVIAME_ENABLE_PYTORCH-NETHARN=ON"
  "-DVIAME_ENABLE_PYTORCH-PYSOT=ON"
  "-DVIAME_BUILD_KWIVER_DIR=C:/tmp/kv1"
  "-DVIAME_BUILD_PLUGINS_DIR=C:/tmp/vm1"
)

set(platform Windows7)
