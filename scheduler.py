import subprocess
import datetime

def run_analysis():
    print(f"Running analysis at {datetime.datetime.now()}")
    subprocess.run(["python", "analyzer.py"])
    
if __name__ == "__main__":
    run_analysis()
