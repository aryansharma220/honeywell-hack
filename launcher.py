"""
Anomaly Detection System Launcher
"""

import sys
import subprocess
import webbrowser
import os
from datetime import datetime


def print_banner():
    """Print an attractive banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║            TIME SERIES ANOMALY DETECTION SYSTEM                   ║
    ║                                                                   ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """Print the main menu"""
    menu = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                           MAIN MENU                             │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Demo (with visualizations)                                  │
    │  2. Streamlit Dashboard (web interface)                         │
    │  3. Command Line Analysis                                       │
    │  4. Exit                                                        │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """
    print(menu)


def run_enhanced_demo():
    """Run the demo with visualizations"""
    print("\nStarting Demo...")
    print("=" * 60)
    
    try:
        import demo
        demo.demonstrate_anomaly_detection()
        
        print("\nDemo completed successfully!")
        print("\nGenerated files:")
        files = [
            "anomaly_results_demo.csv",
            "anomaly_timeseries.png", 
            "feature_importance.png",
            "advanced_analytics.png",
            "interactive_dashboard.html"
        ]
        
        for file in files:
            if os.path.exists(file):
                print(f"   • {file}")
        
                
    except Exception as e:
        print(f"Error running demo: {e}")
        print("Try running option 7 to install dependencies first")


def run_streamlit_dashboard():
    """Launch the Streamlit dashboard"""
    print("\nStarting Streamlit Dashboard...")
    print("=" * 60)
    print("Dashboard will open at: http://localhost:8501")
    print("Use Ctrl+C to stop the server")
    print()
    
    try:
        # Check if streamlit is installed
        subprocess.run([sys.executable, "-c", "import streamlit"], check=True, 
                      capture_output=True)
        
        # Launch streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
        
    except subprocess.CalledProcessError:
        print("Streamlit not installed!")
        print("Run option 7 to install dependencies first")
    except FileNotFoundError:
        print("dashboard.py not found!")
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error launching dashboard: {e}")


def run_cli_analysis():
    """Run command line analysis"""
    print("\nCommand Line Analysis")
    print("=" * 60)
    
    # Get input file
    input_file = input("Enter input CSV file path (or press Enter for sample_dataset.csv): ").strip()
    if not input_file:
        input_file = "sample_dataset.csv"
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    # Get output file
    output_file = input("Enter output CSV file path (or press Enter for auto-generated): ").strip()
    if not output_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"anomaly_results_{timestamp}.csv"
    
    print(f"\nProcessing {input_file}...")
    
    try:
        from anomaly_detector import detect_anomalies
        detect_anomalies(input_file, output_file)
        
        print(f"Analysis completed!")
        print(f"Results saved to: {output_file}")
        
        # Offer to generate visualizations
        choice = input("\nGenerate visualizations? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            print("Please use demo.py for visualizations")
            
    except Exception as e:
        print(f"Error during analysis: {e}")


def main():
    """Main launcher function"""
    while True:
        print_banner()
        print_menu()
        
        try:
            choice = input("Choose an option (1-4): ").strip()
            
            if choice == "1":
                run_enhanced_demo()
            elif choice == "2":
                run_streamlit_dashboard()
            elif choice == "3":
                run_cli_analysis()
            elif choice == "4":
                print("\nThank you for using the Anomaly Detection System!")
                print("Have a great day!")
                break
            else:
                print("\nInvalid choice. Please select 1-4.")
            
            # Pause before showing menu again
            if choice != "4":
                input("\nPress Enter to continue...")
                print("\n" * 2)  # Clear screen effect
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
