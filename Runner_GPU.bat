@echo off
echo Start testing GPU filtering

for %%i in (1 5 25 50 100) do (
    echo Current number of pictures: %%i
    python GPU.py 5 %%i
)

echo Batch script completed.