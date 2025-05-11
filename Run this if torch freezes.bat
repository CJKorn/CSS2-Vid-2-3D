@echo off
set "FILE_PATH=%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py313_cu128\deform_attn\lock"

if exist "%FILE_PATH%" (
    del /F /Q "%FILE_PATH%"
    echo File deleted: %FILE_PATH%
) else (
    echo File not found: %FILE_PATH%
)

pause
