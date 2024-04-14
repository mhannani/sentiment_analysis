@echo off

rem Define the name of the environment
set ENV_NAME=sentiment_analysis

rem Activate the environment
call conda activate %ENV_NAME%

rem Export the environment to a YAML file
call conda env export --no-builds --from-history | findstr /V "prefix" > environment.yml

rem Deactivate the environment
call conda deactivate

echo Environment exported to environment.yml