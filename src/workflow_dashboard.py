"""
GitHub Actions Workflow Dashboard
Analyzes and displays information about the IRR workflow
"""
import os
import re
from datetime import datetime
def load_workflow_data():
    """Load and parse the GitHub Actions workflow file manually"""
    workflow_path = ".github/workflows/irr.yml"
    if not os.path.exists(workflow_path):
        return None
    workflow_data = {
        'name': '',
        'on': {},
        'jobs': {}
    }
    with open(workflow_path, 'r') as f:
        content = f.read()
    name_match = re.search(r'^name:\s*(.+)$', content, re.MULTILINE)
    if name_match:
        workflow_data['name'] = name_match.group(1).strip()
    on_section = re.search(r'^on:\s*\n((?:  .+\n?)*)', content, re.MULTILINE)
    if on_section:
        triggers = {}
        trigger_lines = on_section.group(1)
        for line in trigger_lines.split('\n'):
            line = line.strip()
            if line and ':' in line:
                trigger = line.split(':')[0].strip()
                triggers[trigger] = None
        workflow_data['on'] = triggers
    jobs_match = re.search(r'^jobs:\s*\n(.*)', content, re.MULTILINE | re.DOTALL)
    if jobs_match:
        workflow_data['jobs'] = {'build': {'runs-on': 'ubuntu-latest'}}
    return workflow_data
def display_header():
    """Display dashboard header"""
    print("=" * 80)
    print("🚀 GITHUB ACTIONS WORKFLOW DASHBOARD")
    print("=" * 80)
    print(f"📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
def display_workflow_overview(workflow_data):
    """Display workflow overview section"""
    print("📋 WORKFLOW OVERVIEW")
    print("-" * 40)
    print(f"Name: {workflow_data.get('name', 'N/A')}")
    triggers = workflow_data.get('on', {})
    trigger_list = list(triggers.keys()) if triggers else []
    print(f"Triggers: {', '.join(trigger_list) if trigger_list else 'None'}")
    print(f"Total Jobs: {len(workflow_data.get('jobs', {}))}")
    print()
def display_workflow_purpose():
    """Display workflow purpose based on project context"""
    print("🎯 WORKFLOW PURPOSE")
    print("-" * 40)
    print("This workflow automates the IRR (Internal Rate of Return) calculation process:")
    print()
    print("• Fetches wage and bonus data from Japan's e-Stat API")
    print("• Downloads global stock index data (ACWI) from Yahoo Finance")
    print("• Calculates IRR for different investment scenarios")
    print("• Simulates monthly investments of wage + bonus/12 for various industries")
    print("• Produces analysis results for asset formation disparity research")
    print()
    print("Industries Analyzed:")
    print("  - Overall average (all industries)")
    print("  - Financial services")
    print("  - Mining")
    print("  - Information & communications")
    print("  - Manufacturing")
    print()
def display_triggers_detail():
    """Display detailed trigger information"""
    print("🎯 TRIGGER CONFIGURATION")
    print("-" * 40)
    print("• Push Events: Triggers on any push to the repository")
    print("• Manual Dispatch: Can be triggered manually from GitHub UI")
    print()
def display_jobs_detail():
    """Display detailed job information"""
    print("⚙️ JOB DETAILS")
    print("-" * 40)
    print("Job: build")
    print("  Runner: ubuntu-latest")
    print("  Steps: 5")
    print()
def display_steps_detail():
    """Display detailed step information"""
    print("📝 STEP-BY-STEP BREAKDOWN")
    print("-" * 40)
    steps = [
        {
            'name': 'Checkout Code',
            'type': 'action',
            'action': 'actions/checkout@v4',
            'description': 'Checks out the repository code'
        },
        {
            'name': 'Setup Python',
            'type': 'action',
            'action': 'actions/setup-python@v4',
            'description': 'Sets up Python 3.x environment',
            'params': {'python-version': '3.x'}
        },
        {
            'name': 'Install Dependencies',
            'type': 'run',
            'command': 'pip install -r requirements.txt',
            'description': 'Installs Python dependencies from requirements.txt'
        },
        {
            'name': 'Run IRR Calculation',
            'type': 'run',
            'command': 'python src/fetch_and_compute_irr.py',
            'description': 'Executes the main IRR calculation script',
            'env': {'ESTAT_APP_ID': '***SECRET***'}
        },
        {
            'name': 'Upload Results',
            'type': 'action',
            'action': 'actions/upload-artifact@v3',
            'description': 'Uploads the IRR calculation results as an artifact',
            'params': {'name': 'irr-results', 'path': 'data/processed/irr_results.csv'}
        }
    ]
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step['name']}")
        print(f"     Description: {step['description']}")
        if step['type'] == 'action':
            print(f"     Action: {step['action']}")
            if 'params' in step:
                print(f"     Parameters:")
                for key, value in step['params'].items():
                    print(f"       • {key}: {value}")
        elif step['type'] == 'run':
            print(f"     Command: {step['command']}")
            if 'env' in step:
                print(f"     Environment:")
                for key, value in step['env'].items():
                    print(f"       • {key}: {value}")
        print()
def display_secrets_and_env():
    """Display environment variables and secrets information"""
    print("🔐 ENVIRONMENT & SECRETS")
    print("-" * 40)
    print("Secrets Used:")
    print("  • ESTAT_APP_ID: Required for accessing Japan's e-Stat API")
    print("    (Must be configured in GitHub repository secrets)")
    print()
def display_artifacts():
    """Display artifacts information"""
    print("📦 ARTIFACTS & OUTPUTS")
    print("-" * 40)
    print("Artifacts Produced:")
    print("  • Name: irr-results")
    print("    Path: data/processed/irr_results.csv")
    print("    Description: CSV file containing IRR calculation results")
    print("    Content: Analysis of investment returns for different industries")
    print("             and starting years from 2004-2016")
    print()
def display_workflow_flow():
    """Display workflow execution flow"""
    print("🔄 WORKFLOW EXECUTION FLOW")
    print("-" * 40)
    print("1. Trigger → Push to repository or manual dispatch")
    print("2. Environment → Ubuntu latest with Python 3.x")
    print("3. Setup → Checkout code and install dependencies")
    print("4. Data Collection → Fetch wage data (e-Stat) and stock data (Yahoo)")
    print("5. Processing → Calculate IRR for multiple scenarios")
    print("6. Output → Generate CSV results and upload as artifact")
    print("7. Completion → Results available for download")
    print()
def display_technical_details():
    """Display technical implementation details"""
    print("🔧 TECHNICAL DETAILS")
    print("-" * 40)
    print("Data Sources:")
    print("  • Japan e-Stat API: Wage and bonus data by industry")
    print("  • Yahoo Finance: ACWI global stock index historical data")
    print()
    print("Calculation Method:")
    print("  • Simulates monthly investments of (wage + bonus/12)")
    print("  • Uses binary search algorithm for IRR calculation")
    print("  • Analyzes investment periods from 2004-2016")
    print()
    print("Output Format:")
    print("  • CSV file with IRR results by industry and start year")
    print("  • Currently outputs CSV (will migrate to YAML per requirements)")
    print()
def display_footer():
    """Display dashboard footer"""
    print("=" * 80)
    print("📊 Dashboard generated for IRR calculation workflow analysis")
    print("🔍 For more details, check the workflow file: .github/workflows/irr.yml")
    print("📋 Project requirements documented in: docs/tasks.md")
    print("=" * 80)
def main():
    """Main dashboard function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    workflow_data = load_workflow_data()
    if not workflow_data:
        print("❌ Error: Could not load workflow file!")
        return
    display_header()
    display_workflow_overview(workflow_data)
    display_workflow_purpose()
    display_triggers_detail()
    display_jobs_detail()
    display_steps_detail()
    display_secrets_and_env()
    display_artifacts()
    display_workflow_flow()
    display_technical_details()
    display_footer()
if __name__ == "__main__":
    main()