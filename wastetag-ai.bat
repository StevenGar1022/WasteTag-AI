@echo off
REM WASTETAG-AI — lanzador Windows. Arranca el CLI sin activar el entorno.
REM Uso (desde la carpeta del repo):  wastetag-ai.bat [flags]
setlocal
set ROOT=%~dp0
if not exist "%ROOT%env\Scripts\python.exe" (
    echo No existe env\. Ejecuta primero:  py install.py
    exit /b 1
)
"%ROOT%env\Scripts\python.exe" "%ROOT%src\wastetag_cli.py" %*
