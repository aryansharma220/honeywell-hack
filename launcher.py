"""
🚀 Anomaly Detection System Launcher
Interactive launcher for all UI components and analysis tools
"""

import sys
import subprocess
import webbrowser
import time
import os
from datetime import datetime
import threading


def print_banner():
    """Print an attractive banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║           🔍 TIME SERIES ANOMALY DETECTION SYSTEM 🔍             ║
    ║                                                                   ║
    ║                     Enhanced UI Launcher v2.0                     ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """Print the main menu"""
    menu = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                          🎛️ MAIN MENU                           │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. 🚀 Enhanced Demo (with visualizations)                      │
    │  2. 🌐 Streamlit Dashboard (web interface)                       │
    │  3.  Command Line Analysis                                    │
    │  4. ❌ Exit                                                     │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """
    print(menu)


def run_enhanced_demo():
    """Run the enhanced demo with visualizations"""
    print("\n🚀 Starting Enhanced Demo...")
    print("=" * 60)
    
    try:
        import demo
        demo.demonstrate_anomaly_detection()
        
        print("\n✅ Demo completed successfully!")
        print("\n📁 Generated files:")
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
        
        while True:
            choice = input("\n🖼️  Open generated visualizations? (y/n): ").lower().strip()
            if choice in ['y', 'yes']:
                try:
                    if os.path.exists("interactive_dashboard.html"):
                        webbrowser.open("interactive_dashboard.html")
                        print("🌐 Interactive dashboard opened in browser")
                except Exception as e:
                    print(f"❌ Could not open browser: {e}")
                break
            elif choice in ['n', 'no']:
                break
            else:
                print("Please enter 'y' or 'n'")
                
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("💡 Try running option 7 to install dependencies first")


def run_streamlit_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n🌐 Starting Streamlit Dashboard...")
    print("=" * 60)
    print("🔗 Dashboard will open at: http://localhost:8501")
    print("📱 Use Ctrl+C to stop the server")
    print()
    
    try:
        # Check if streamlit is installed
        subprocess.run([sys.executable, "-c", "import streamlit"], check=True, 
                      capture_output=True)
        
        # Launch streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
        
    except subprocess.CalledProcessError:
        print("❌ Streamlit not installed!")
        print("💡 Run option 7 to install dependencies first")
    except FileNotFoundError:
        print("❌ dashboard.py not found!")
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")


def run_cli_analysis():
    """Run command line analysis"""
    print("\n🔧 Command Line Analysis")
    print("=" * 60)
    
    # Get input file
    input_file = input("📁 Enter input CSV file path (or press Enter for sample_dataset.csv): ").strip()
    if not input_file:
        input_file = "sample_dataset.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    # Get output file
    output_file = input("💾 Enter output CSV file path (or press Enter for auto-generated): ").strip()
    if not output_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"anomaly_results_{timestamp}.csv"
    
    print(f"\n🔄 Processing {input_file}...")
    
    try:
        from anomaly_detector import detect_anomalies
        detect_anomalies(input_file, output_file)
        
        print(f"✅ Analysis completed!")
        print(f"📄 Results saved to: {output_file}")
        
        # Offer to generate visualizations
        choice = input("\n📊 Generate visualizations? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            generate_visualizations(output_file)
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")


def generate_visualizations(results_file=None):
    """Generate visualizations for existing results"""
    pass  # Function removed - use demo.py instead


def generate_summary_report():
    """Generate a comprehensive summary report"""
    pass  # Function removed - use demo.py instead


def install_dependencies():
    """Install or update dependencies"""
    pass  # Function removed - use pip directly


def show_documentation():
    """Show documentation and help"""
    doc = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                          📖 DOCUMENTATION                        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  🔍 ANOMALY DETECTION SYSTEM                                     ║
    ║                                                                  ║
    ║  This system detects anomalies in multivariate time series      ║
    ║  data using Isolation Forest algorithm and provides feature     ║
    ║  attribution for each detected anomaly.                         ║
    ║                                                                  ║
    ║  📊 USER INTERFACES:                                             ║
    ║                                                                  ║
    ║  • Enhanced Demo: Rich command-line demo with visualizations    ║
    ║  • Streamlit Dashboard: Web-based interactive interface         ║
    ║  • Real-time Monitor: Live monitoring with alerts              ║
    ║  • Command Line: Traditional CLI analysis                       ║
    ║                                                                  ║
    ║  📋 DATA REQUIREMENTS:                                           ║
    ║                                                                  ║
    ║  • CSV file with 'Time' column (MM/DD/YYYY HH:MM format)       ║
    ║  • Multiple numerical sensor/feature columns                    ║
    ║  • Training period: 1/1/2004 0:00 to 1/5/2004 23:59           ║
    ║  • Analysis period: 1/1/2004 0:00 to 1/19/2004 7:59           ║
    ║                                                                  ║
    ║  🎯 OUTPUT:                                                      ║
    ║                                                                  ║
    ║  • Anomaly scores (0-100 scale)                                 ║
    ║  • Top contributing features for each anomaly                   ║
    ║  • Interactive visualizations and reports                       ║
    ║  • Exportable results in multiple formats                       ║
    ║                                                                  ║
    ║  🔧 ALGORITHM:                                                   ║
    ║                                                                  ║
    ║  • Isolation Forest for unsupervised anomaly detection         ║
    ║  • Statistical feature importance calculation                    ║
    ║  • Standardized preprocessing and validation                     ║
    ║                                                                  ║
    ║  📁 FILES:                                                       ║
    ║                                                                  ║
    ║  • launcher.py: This interactive launcher                       ║
    ║  • demo.py: Enhanced demo with visualizations                   ║
    ║  • dashboard.py: Streamlit web dashboard                        ║
    ║  • realtime_monitor.py: Live monitoring interface              ║
    ║  • anomaly_detector.py: Core detection algorithm               ║
    ║  • utils.py: Utility functions and visualizations              ║
    ║  • config.py: Configuration parameters                          ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    💡 QUICK START TIPS:
    
    1. Start with option 1 (Enhanced Demo) to see all features
    2. Use option 2 (Streamlit Dashboard) for interactive analysis
    3. Try option 3 (Real-time Monitor) for live monitoring simulation
    4. Run option 7 first if you encounter dependency errors
    
    🔗 For more information, check the README.md file
    """
    print(doc)


def main():
    """Main launcher function"""
    while True:
        print_banner()
        print_menu()
        
        try:
            choice = input("🎯 Choose an option (1-4): ").strip()
            
            if choice == "1":
                run_enhanced_demo()
            elif choice == "2":
                run_streamlit_dashboard()
            elif choice == "3":
                run_cli_analysis()
            elif choice == "4":
                print("\n👋 Thank you for using the Anomaly Detection System!")
                print("🚀 Have a great day!")
                break
            else:
                print("\n❌ Invalid choice. Please select 1-4.")
            
            # Pause before showing menu again
            if choice != "4":
                input("\n⏸️  Press Enter to continue...")
                print("\n" * 2)  # Clear screen effect
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("⏸️  Press Enter to continue...")


if __name__ == "__main__":
    main()
