import pytest
import sys
import os
import re

class LoggerWriter:
    """
    A custom writer class that acts like the 'tee' command.
    It writes output to both the terminal (stdout) and a log file simultaneously.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        self.terminal.write(message)
        clean_message = self.ansi_escape.sub('', message)
        self.log.write(clean_message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        """
        Crucial Fix: Pytest checks this to enable colors and progress bars.
        We delegate this check to the actual terminal.
        """
        return getattr(self.terminal, 'isatty', lambda: False)()

def main():
    # --- PATH SETUP ---
    
    # 1. Get the directory where this script is located (i.e., the 'test' folder)
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Get the project root directory (parent of 'test')
    project_root = os.path.dirname(test_dir)
    
    # 3. Add project root to sys.path
    # This ensures that pytest can import modules from 'src' correctly
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # --- LOGGING SETUP ---

    # 4. Define the log directory inside the 'test' folder
    log_dir = os.path.join(test_dir, "log")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")

    log_file_path = os.path.join(log_dir, "test_run.log")

    print(f"Starting test suite... Logs will be saved to: {log_file_path}")
    print("-" * 50)

    # --- EXECUTION ---

    original_stdout = sys.stdout
    sys.stdout = LoggerWriter(log_file_path)

    try:
        # 5. Run pytest
        # We pass 'test_dir' to tell pytest to scan the folder where this script lives.
        exit_code = pytest.main(["-v", test_dir])
    
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
        exit_code = 1
        
    finally:
        sys.stdout = original_stdout
        print("-" * 50)
        
        if exit_code == 0:
            print("Tests completed successfully! ✅")
        else:
            print("Tests failed. Check the log for details. ❌")

    sys.exit(exit_code)

if __name__ == "__main__":
    main()