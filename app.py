import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

# --- Configuration & Tax Data for 2025 ---
TODAY_YEAR = datetime.now().year
FED_BRACKETS_2025 = {57375: 0.15, 114750: 0.205, 177882: 0.26, 253414: 0.29, float('inf'): 0.33}
ON_BRACKETS_2025 = {52886: 0.0505, 105775: 0.0915, 150000: 0.1116, 220000: 0.1216, float('inf'): 0.1316}
FED_BPA_2025 = 16129
ON_BPA_2025 = 12399
OAS_CLAWBACK_THRESHOLD_2025 = 93454
OAS_CLAWBACK_RATE = 0.15
ELIGIBLE_DIVIDEND_GROSS_UP = 1.38
FED_ELIGIBLE_DIVIDEND_CREDIT_RATE = 0.150198
ON_ELIGIBLE_DIVIDEND_CREDIT_RATE = 0.10

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        data = requests.get("https://api.exchangerate-api.com/v4/latest/USD").json()
        return data['rates']['CAD']
    except Exception:
        return 1.35

def format_currency(amount):
    return f"${amount:,.0f}"

# --- Core Tax Calculation Logic (from Net Income Calculator) ---
def calculate_progressive_tax(income, brackets):
    tax = 0
    previous_bracket_limit = 0
    for limit, rate in brackets.items():
        if income > limit:
            taxable_in_bracket = limit - previous_bracket_limit
            tax += taxable_in_bracket * rate
            previous_bracket_limit = limit
        else:
            taxable_in_bracket = income - previous_bracket_limit
            tax += taxable_in_bracket * rate
            break
    return tax

def calculate_after_tax_income(yearly_income_details):
    """
    Calculates the after-tax income for a given year's income details.
    This is the integrated core of the net income calculator.
    """
    gross_income = sum(yearly_income_details.values())
    
    eligible_dividends_actual = yearly_income_details.get('cdn_dividends', 0)
    grossed_up_dividends = eligible_dividends_actual * ELIGIBLE_DIVIDEND_GROSS_UP

    us_dividends_cad = yearly_income_details.get('us_dividends', 0)
    us_dividend_account = st.session_state.get('us_dividend_account', 'Non-Registered')
    
    taxable_us_dividends = 0
    us_withholding_tax = 0
    if us_dividend_account in ['Non-Registered', 'TFSA']:
        us_withholding_tax = us_dividends_cad * 0.15
        if us_dividend_account == 'Non-Registered':
            taxable_us_dividends = us_dividends_cad
    
    capital_gains_taxable = yearly_income_details.get('capital_gains', 0) * 0.5

    total_taxable_income = (
        yearly_income_details.get('cpp_oas', 0) + yearly_income_details.get('pension_rrif', 0) +
        yearly_income_details.get('other_income', 0) + yearly_income_details.get('interest', 0) +
        grossed_up_dividends + taxable_us_dividends + capital_gains_taxable
    )

    federal_tax_before_credits = calculate_progressive_tax(total_taxable_income, FED_BRACKETS_2025)
    provincial_tax_before_credits = calculate_progressive_tax(total_taxable_income, ON_BRACKETS_2025)
    
    federal_tax = federal_tax_before_credits
    provincial_tax = provincial_tax_before_credits

    federal_tax -= min(total_taxable_income, FED_BPA_2025) * list(FED_BRACKETS_2025.values())[0]
    provincial_tax -= min(total_taxable_income, ON_BPA_2025) * list(ON_BRACKETS_2025.values())[0]

    federal_tax -= grossed_up_dividends * FED_ELIGIBLE_DIVIDEND_CREDIT_RATE
    provincial_tax -= grossed_up_dividends * ON_ELIGIBLE_DIVIDEND_CREDIT_RATE

    if us_dividend_account == 'Non-Registered' and total_taxable_income > 0:
        canadian_tax_on_us_income = taxable_us_dividends * (federal_tax_before_credits / total_taxable_income)
        foreign_tax_credit = min(us_withholding_tax, canadian_tax_on_us_income)
        federal_tax -= foreign_tax_credit

    federal_tax = max(0, federal_tax)
    provincial_tax = max(0, provincial_tax)

    net_income_for_clawback = total_taxable_income - (grossed_up_dividends - eligible_dividends_actual)
    oas_clawback = 0
    if net_income_for_clawback > OAS_CLAWBACK_THRESHOLD_2025:
        oas_clawback = (net_income_for_clawback - OAS_CLAWBACK_THRESHOLD_2025) * OAS_CLAWBACK_RATE
        oas_clawback = min(oas_clawback, yearly_income_details.get('cpp_oas', 0))

    total_tax = federal_tax + provincial_tax + oas_clawback
    net_income = gross_income - total_tax
    return net_income

# --- Core Retirement Simulation Engine ---
def run_simulation(scenario):
    investment = float(scenario['initialInvestment'])
    yearly_data = []
    
    for year in range(TODAY_YEAR, scenario['endYear'] + 1):
        start_balance = investment
        
        # Calculate Future Value of all incomes for the current year
        yearly_income_details = {}
        for income in scenario['incomes']:
            if year >= int(income['startYear']):
                years_active = year - int(income['startYear'])
                future_value = float(income['amount']) * ((1 + float(income['growthRate']) / 100) ** years_active)
                
                # Categorize income for tax calculation
                income_type = 'other_income' # default
                if 'cpp' in income['name'].lower() or 'oas' in income['name'].lower():
                    income_type = 'cpp_oas'
                elif 'pension' in income['name'].lower() or 'rrif' in income['name'].lower():
                    income_type = 'pension_rrif'
                
                yearly_income_details[income_type] = yearly_income_details.get(income_type, 0) + future_value

        # Calculate this year's after-tax income using the integrated function
        net_annual_income = calculate_after_tax_income(yearly_income_details)

        # Calculate Future Value of expenses
        net_annual_expense = sum(
            float(exp['amount']) * ((1 + float(exp['growthRate']) / 100) ** (year - TODAY_YEAR))
            for exp in scenario['expenses']
        )
        
        # Apply net income/expense
        investment += (net_annual_income - net_annual_expense)
        
        # Apply investment return
        if investment > 0:
            investment *= (1 + float(scenario['investmentReturn']) / 100)

        yearly_data.append({'year': year, 'balance': round(investment)})
        if investment <= 0:
            break
            
    return yearly_data

# --- UI Components ---
def income_input_ui(key_suffix):
    st.markdown("##### Recurring Incomes")
    if 'incomes' not in st.session_state:
        st.session_state.incomes = []

    for i, income in enumerate(st.session_state.incomes):
        cols = st.columns([3, 2, 2, 2, 1])
        income['name'] = cols[0].text_input("Name", value=income['name'], key=f"in_name_{i}_{key_suffix}")
        income['amount'] = cols[1].number_input("Amount (Today's $)", value=income['amount'], key=f"in_amount_{i}_{key_suffix}")
        income['startYear'] = cols[2].number_input("Start Year", value=income['startYear'], key=f"in_start_{i}_{key_suffix}")
        income['growthRate'] = cols[3].number_input("Growth (%)", value=income['growthRate'], key=f"in_growth_{i}_{key_suffix}")
        if cols[4].button("🗑️", key=f"in_del_{i}_{key_suffix}"):
            st.session_state.incomes.pop(i)
            st.rerun() # BUG FIX: Replaced st.experimental_rerun()

    if st.button("Add Income", key=f"add_income_{key_suffix}"):
        st.session_state.incomes.append({'name': 'New Income', 'amount': 10000, 'startYear': TODAY_YEAR + 10, 'growthRate': 2.5})
        st.rerun() # BUG FIX: Replaced st.experimental_rerun()

def expense_input_ui(key_suffix):
    st.markdown("##### Recurring Expenses")
    if 'expenses' not in st.session_state:
        st.session_state.expenses = []

    for i, exp in enumerate(st.session_state.expenses):
        cols = st.columns([5, 3, 3, 1])
        exp['name'] = cols[0].text_input("Name", value=exp['name'], key=f"ex_name_{i}_{key_suffix}")
        exp['amount'] = cols[1].number_input("Amount (Today's $)", value=exp['amount'], key=f"ex_amount_{i}_{key_suffix}")
        exp['growthRate'] = cols[2].number_input("Growth (%)", value=exp['growthRate'], key=f"ex_growth_{i}_{key_suffix}")
        if cols[3].button("🗑️", key=f"ex_del_{i}_{key_suffix}"):
            st.session_state.expenses.pop(i)
            st.rerun() # BUG FIX: Replaced st.experimental_rerun()

    if st.button("Add Expense", key=f"add_expense_{key_suffix}"):
        st.session_state.expenses.append({'name': 'Living Expenses', 'amount': 50000, 'growthRate': 3})
        st.rerun() # BUG FIX: Replaced st.experimental_rerun()

# --- Main App Layout ---
st.set_page_config(layout="wide")
st.title("통합 은퇴 및 세후 소득 계산기")
st.markdown("장기 은퇴 계획과 연간 세금 계산을 결합하여 정확한 자산 변화를 시뮬레이션합니다.")

# Initialize session state
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = [{
        'name': 'My Retirement Plan',
        'initialInvestment': 500000, 'investmentReturn': 6, 'endYear': TODAY_YEAR + 40,
        'incomes': [{'name': 'CPP/OAS', 'amount': 15000, 'startYear': TODAY_YEAR + 25, 'growthRate': 2.5}],
        'expenses': [{'name': 'Base Expenses', 'amount': 60000, 'growthRate': 3.0}]
    }]

# Sidebar for scenario management and inputs
with st.sidebar:
    st.header("⚙️ Inputs & Settings")
    
    scenario_names = [s['name'] for s in st.session_state.scenarios]
    selected_scenario_name = st.selectbox("Select Scenario", scenario_names)
    active_scenario_index = scenario_names.index(selected_scenario_name)
    
    st.markdown("---")
    
    # Scenario details
    st.session_state.scenarios[active_scenario_index]['name'] = st.text_input(
        "Scenario Name", value=st.session_state.scenarios[active_scenario_index]['name']
    )
    
    cols = st.columns(2)
    st.session_state.scenarios[active_scenario_index]['initialInvestment'] = cols[0].number_input("Initial Investments", value=st.session_state.scenarios[active_scenario_index]['initialInvestment'])
    st.session_state.scenarios[active_scenario_index]['investmentReturn'] = cols[1].number_input("Avg. Return (%)", value=st.session_state.scenarios[active_scenario_index]['investmentReturn'])
    st.session_state.scenarios[active_scenario_index]['endYear'] = st.slider("Retirement End Year", TODAY_YEAR, TODAY_YEAR + 60, st.session_state.scenarios[active_scenario_index]['endYear'])

    st.session_state.us_dividend_account = st.selectbox(
        "US Dividend Account Type", 
        ("Non-Registered", "RRSP/RRIF", "TFSA"),
        help="모든 미국 배당금에 적용될 계좌 유형을 선택하세요."
    )

    with st.expander("Income & Expenses", expanded=True):
        # Use session state directly for incomes/expenses to persist them
        if 'incomes' not in st.session_state:
            st.session_state.incomes = st.session_state.scenarios[active_scenario_index]['incomes']
        if 'expenses' not in st.session_state:
            st.session_state.expenses = st.session_state.scenarios[active_scenario_index]['expenses']
            
        income_input_ui(active_scenario_index)
        st.markdown("---")
        expense_input_ui(active_scenario_index)

        # Sync back to the main scenario object before simulation
        st.session_state.scenarios[active_scenario_index]['incomes'] = st.session_state.incomes
        st.session_state.scenarios[active_scenario_index]['expenses'] = st.session_state.expenses

# Main panel for results
st.header("📊 Simulation Results")

if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Calculating..."):
        all_results = [run_simulation(s) for s in st.session_state.scenarios]

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, result_data in enumerate(all_results):
        if result_data:
            years = [d['year'] for d in result_data]
            balances = [d['balance'] for d in result_data]
            fig.add_trace(go.Scatter(
                x=years, y=balances,
                mode='lines',
                name=st.session_state.scenarios[i]['name'],
                line=dict(color=colors[i % len(colors)], width=3),
                fill='tozeroy',
            ))

    fig.update_layout(
        title="Retirement Portfolio Projection",
        xaxis_title="Year",
        yaxis_title="Portfolio Balance",
        yaxis_tickprefix="$",
        yaxis_tickformat="~s",
        legend_title="Scenarios",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("입력값을 조정한 후 'Run Simulation' 버튼을 눌러주세요.")
