@echo off
echo Start testing Sequential filtering
:: Loop though the values (1, 5, 25 ,50, 100)
for %%i in (1 5 25 50 100) do (
    echo Current number of pictures: %%i
    :: run python and pass the kernel size = 5
    python Sequentialprocessing.py 5 %%i
)

echo Batch script completed.
