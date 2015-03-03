SET ENVDIR=pygeoprocessing_virtualenv
python bootstrap_invest_environment.py > setup_environment.py

IF "%1"== "clean" GOTO CLEANBUILD

python setup_environment.py --system-site-packages %ENVDIR%
GOTO BULIDDONE

:CLEANBUILD
DEL /S /Q build
DEL /S /Q dist
DEL /S /Q pygeoprocessing.egg-info
DEL /S /Q %ENVDIR%
python setup_environment.py --clear --system-site-packages %ENVDIR%

GOTO BULIDDONE

:BULIDDONE
copy C:\Python27\Lib\distutils\distutils.cfg .\%ENVDIR%\Lib\distutils\distutils.cfg
%ENVDIR%\Scripts\python setup.py install
%ENVDIR%\Scripts\python pygeoprocessing\tests\test_suite.py
