@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=C:\Program Files\VIAME

REM Processing options
SET INPUT_DIRECTORY=videos
SET DEFAULT_FRAME_RATE=5
SET MAX_DURATION=05:00.00
SET OUTPUT_DIR=video_clips

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

python.exe "%VIAME_INSTALL%\configs\process_videos.py" ^
  -p "pipelines/transcode_default.pipe" ^
  -d "%INPUT_DIRECTORY%" -o %OUTPUT_DIR% ^
  -r %DEFAULT_FRAME_RATE% ^
  -s "output:maximum_length=%MAX_DURATION%"

pause
