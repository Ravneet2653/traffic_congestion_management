import subprocess
import sys
import os

def run_step(command, description):
    """Run a command with description"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"❌ Error in {description}")
        return False
    
    print(f"✅ {description} completed!")
    return True

def main():
    print("\n" + "="*60)
    print("🚦 AI TRAFFIC CONTROL SYSTEM - COMPLETE SETUP")
    print("="*60)
    
    
    steps = [
        ("python generate_data.py", "Step 1: Generating Data"),
        ("python preprocess_data.py", "Step 2: Preprocessing Data"),
        ("python train_models.py", "Step 3: Training Models"),
    ]
    
    for command, description in steps:
        if not run_step(command, description):
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ All steps completed successfully!")
    print("="*60)
    print("\n🎮 Now run the simulation:")
    print("   cd TrafficLightSimulation")
    print("   python main.py")
    print("\nOr press Enter to run now...")
    
    input()
    
    os.chdir('TrafficLightSimulation')
    subprocess.run("python main.py", shell=True)

if __name__ == "__main__":
    main()