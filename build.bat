@echo off
nvcc -o lab.exe wmma_lab.cu -arch=sm_120 -O3 -lcublas
if errorlevel 1 (
    echo [FAIL] Compilation failed.
    exit /b 1
)
echo [OK] Compiled successfully. Running...
lab.exe
if errorlevel 1 (
    echo [FAIL] Verification failed.
    exit /b 1
)
