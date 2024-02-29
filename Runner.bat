@echo off
echo Start testing Sequential filtering

for %%i in (1) do (
    echo Current number of pictures: %%i
    python Sequentialprocessing.py 5 %%i
)

echo Batch script completed.
