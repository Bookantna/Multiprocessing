@echo off
echo Start testing multiprocessing filtering

for %%i in (1) do (
    echo Current number of pictures: %%i
    python Multiprocessing.py 5 %%i
)

echo Batch script completed.
