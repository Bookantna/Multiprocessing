@echo off
echo Start testing GPU filtering

for %%i in (1) do (
    echo Current number of pictures: %%i
    python Main.py 5 %%i
)

echo Batch script completed.