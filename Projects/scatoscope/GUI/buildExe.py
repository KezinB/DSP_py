from cx_Freeze import setup, Executable

setup(
    name="MyApp",
    version="1.0",
    description="My Python Application",
    executables=[Executable("guiMain.py")]
)
