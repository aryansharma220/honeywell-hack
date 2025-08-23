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
    │  3. 📊 Real-time Monitor (live dashboard)                       │
    │  4. 🔧 Command Line Analysis                                    │
    │  5. 📸 Generate Visualizations Only                             │
    │  6. 📄 Generate Summary Report                                  │
    │  7. 🛠️ Install/Update Dependencies                              │
    │  8. 📖 View Documentation                                       │
    │  9. ❌ Exit                                                     │
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


def run_realtime_monitor():
    """Launch the real-time monitoring dashboard"""
    print("\n📊 Starting Real-time Monitor...")
    print("=" * 60)
    print("🔗 Monitor will open at: http://localhost:8050")
    print("📱 Use Ctrl+C to stop the server")
    print()
    
    try:
        # Check if dash is installed
        subprocess.run([sys.executable, "-c", "import dash"], check=True, 
                      capture_output=True)
        
        # Launch monitor
        subprocess.run([sys.executable, "realtime_monitor.py"])
        
    except subprocess.CalledProcessError:
        print("❌ Dash not installed!")
        print("💡 Run option 7 to install dependencies first")
    except FileNotFoundError:
        print("❌ realtime_monitor.py not found!")
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped by user")
    except Exception as e:
        print(f"❌ Error launching monitor: {e}")


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
    print("\n📸 Generating Visualizations...")
    print("=" * 60)
    
    if not results_file:
        results_file = input("📁 Enter results CSV file path: ").strip()
    
    if not os.path.exists(results_file):
        print(f"❌ File not found: {results_file}")
        return
    
    try:
        import pandas as pd
        import utils
        
        print(f"📊 Loading data from {results_file}...")
        df = pd.read_csv(results_file)
        
        print("🎨 Creating visualizations...")
        utils.create_quick_visualization(df, save_plots=True)
        
        print("✅ Visualizations generated successfully!")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")


def generate_summary_report():
    """Generate a comprehensive summary report"""
    print("\n📄 Generating Summary Report...")
    print("=" * 60)
    
    results_file = input("📁 Enter results CSV file path: ").strip()
    
    if not os.path.exists(results_file):
        print(f"❌ File not found: {results_file}")
        return
    
    try:
        import pandas as pd
        import utils
        
        print(f"📊 Loading data from {results_file}...")
        df = pd.read_csv(results_file)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"anomaly_report_{timestamp}.txt"
        
        print("📝 Generating report...")
        report = utils.generate_summary_report(df, report_file)
        
        print(report)
        print(f"\n📄 Report saved to: {report_file}")
        
        # Offer export options
        print("\n💾 Export Options:")
        print("1. JSON format")
        print("2. Excel format") 
        print("3. Skip export")
        
        choice = input("Choose export format (1-3): ").strip()
        
        if choice == "1":
            utils.export_data_for_external_tools(df, "json")
        elif choice == "2":
            utils.export_data_for_external_tools(df, "excel")
        elif choice == "3":
            pass
        else:
            print("Invalid choice, skipping export")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")


def install_dependencies():
    """Install or update dependencies"""
    print("\n🛠️ Installing/Updating Dependencies...")
    print("=" * 60)
    
    try:
        print("📦 Updating pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True)
        
        print("📦 Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        
        print("✅ Dependencies installed successfully!")
        
        # Verify key packages
        print("\n🔍 Verifying installation...")
        packages = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", 
                   "streamlit", "plotly", "dash"]
        
        for package in packages:
            try:
                subprocess.run([sys.executable, "-c", f"import {package}"], 
                             check=True, capture_output=True)
                print(f"   ✅ {package}")
            except:
                print(f"   ❌ {package}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
    except FileNotFoundError:
        print("❌ requirements.txt not found!")


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
            choice = input("🎯 Choose an option (1-9): ").strip()
            
            if choice == "1":
                run_enhanced_demo()
            elif choice == "2":
                run_streamlit_dashboard()
            elif choice == "3":
                run_realtime_monitor()
            elif choice == "4":
                run_cli_analysis()
            elif choice == "5":
                generate_visualizations()
            elif choice == "6":
                generate_summary_report()
            elif choice == "7":
                install_dependencies()
            elif choice == "8":
                show_documentation()
            elif choice == "9":
                print("\n👋 Thank you for using the Anomaly Detection System!")
                print("🚀 Have a great day!")
                break
            else:
                print("\n❌ Invalid choice. Please select 1-9.")
            
            # Pause before showing menu again
            if choice != "9":
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
