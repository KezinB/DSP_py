from cx_Freeze import setup, Executable

# Define the executable and options
executables = [
    Executable(
        r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\mockInterview\codes\UI\Qt\test2.py",
        base="Win32GUI",  # Use Win32GUI to hide the console for GUI apps
        target_name="YourAppName.exe",  # Name of the output executable
        icon="icon.ico",  # Optional: Add your icon file here
    )
]

# Setup configuration
setup(
    name="YourAppName",
    version="1.0",
    description="Your Application Description",
    executables=executables,
)
