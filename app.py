import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime
import copy

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

# --- Core Tax Calculation Logic ---
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

def calculate_after_tax_income(yearly_income_details, us_dividend_account):
    gross_income = sum(yearly_income_details.values())
    eligible_dividends_actual = yearly_income_details.get('cdn_dividends', 0)
    grossed_up_dividends = eligible_dividends_actual * ELIGIBLE_DIVIDEND_GROSS_UP
    us_dividends_cad = yearly_income_details.get('us_dividends', 0)
    
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
    
    federal_tax, provincial_tax = federal_tax_before_credits, provincial_tax_before_credits

    federal_tax -= min(total_taxable_income, FED_BPA_2025) * list(FED_BRACKETS_2025.values())[0]
    provincial_tax -= min(total_taxable_income, ON_BPA_2025) * list(ON_BRACKETS_2025.values())[0]

    federal_tax -= grossed_up_dividends * FED_ELIGIBLE_DIVIDEND_CREDIT_RATE
    provincial_tax -= grossed_up_dividends * ON_ELIGIBLE_DIVIDEND_CREDIT_RATE

    if us_dividend_account == 'Non-Registered' and total_taxable_income > 0:
        canadian_tax_on_us_income = taxable_us_dividends * (federal_tax_before_credits / total_taxable_income)
        foreign_tax_credit = min(us_withholding_tax, canadian_tax_on_us_income)
        federal_tax -= foreign_tax_credit

    federal_tax, provincial_tax = max(0, federal_tax), max(0, provincial_tax)

    net_income_for_clawback = total_taxable_income - (grossed_up_dividends - eligible_dividends_actual)
    oas_clawback = 0
    if net_income_for_clawback > OAS_CLAWBACK_THRESHOLD_2025:
        oas_clawback = (net_income_for_clawback - OAS_CLAWBACK_THRESHOLD_2025) * OAS_CLAWBACK_RATE
        oas_clawback = min(oas_clawback, yearly_income_details.get('cpp_oas', 0))

    total_tax = federal_tax + provincial_tax + oas_clawback
    net_income = gross_income - total_tax
    return net_income

# --- Core Retirement Simulation Engine ---
def run_simulation(scenario, exchange_rate):
    investment = float(scenario['initialInvestment'])
    yearly_data = []
    depletion_year = None

    for year in range(TODAY_YEAR, scenario['endYear'] + 1):
        if investment <= 0 and depletion_year is None:
            depletion_year = year
        
        balance = max(0, investment)
        age = year - scenario['birthYear']
        yearly_data.append({'year': year, 'age': age, 'balance': round(balance)})

        # Calculate Future Value of all incomes
        yearly_income_details = {}
        for income in scenario['incomes']:
            if year >= int(income['startYear']):
                years_from_today = int(income['startYear']) - TODAY_YEAR
                amount_at_start = float(income['amount']) * ((1 + float(income['growthRate']) / 100) ** (years_from_today if years_from_today > 0 else 0))
                
                years_since_start = year - int(income['startYear'])
                future_value = amount_at_start * ((1 + float(income['growthRate']) / 100) ** years_since_start)

                income_type = income.get('type', 'other_income')
                if income_type == 'us_dividends':
                    future_value *= exchange_rate

                yearly_income_details[income_type] = yearly_income_details.get(income_type, 0) + future_value
        
        net_annual_income = calculate_after_tax_income(yearly_income_details, scenario['us_dividend_account'])

        # Calculate Future Value of expenses
        net_annual_expense = sum(
            float(exp['amount']) * ((1 + float(exp['growthRate']) / 100) ** (year - TODAY_YEAR))
            for exp in scenario['expenses']
        )
        
        investment += (net_annual_income - net_annual_expense)
        
        if investment > 0:
            investment *= (1 + float(scenario['investmentReturn']) / 100)
            
    return {'data': yearly_data, 'depletion_year': depletion_year}

# --- UI Components ---
def income_input_ui(key_suffix):
    st.markdown("<h6>Recurring Incomes</h6>", unsafe_allow_html=True)
    active_scenario = st.session_state.scenarios[st.session_state.active_scenario_index]

    for i, income in enumerate(active_scenario['incomes']):
        cols = st.columns([3, 2, 2, 2, 1])
        income['name'] = cols[0].text_input("Name", value=income['name'], key=f"in_name_{i}_{key_suffix}")
        income['amount'] = cols[1].number_input("Amount (Today's $)", value=income['amount'], key=f"in_amount_{i}_{key_suffix}")
        income['startYear'] = cols[2].number_input("Start Year", value=income['startYear'], min_value=TODAY_YEAR, key=f"in_start_{i}_{key_suffix}")
        income['growthRate'] = cols[3].number_input("Growth (%)", value=income['growthRate'], key=f"in_growth_{i}_{key_suffix}")
        if cols[4].button("🗑️", key=f"in_del_{i}_{key_suffix}", help="Remove this income item"):
            active_scenario['incomes'].pop(i)
            st.rerun()

    if st.button("Add Income", key=f"add_income_{key_suffix}"):
        active_scenario['incomes'].append({'name': 'New Income', 'amount': 10000, 'startYear': TODAY_YEAR + 10, 'growthRate': 2.5, 'type': 'other_income'})
        st.rerun()

def expense_input_ui(key_suffix):
    st.markdown("<h6>Recurring Expenses</h6>", unsafe_allow_html=True)
    active_scenario = st.session_state.scenarios[st.session_state.active_scenario_index]

    for i, exp in enumerate(active_scenario['expenses']):
        cols = st.columns([5, 3, 3, 1])
        exp['name'] = cols[0].text_input("Name", value=exp['name'], key=f"ex_name_{i}_{key_suffix}")
        exp['amount'] = cols[1].number_input("Amount (Today's $)", value=exp['amount'], key=f"ex_amount_{i}_{key_suffix}")
        exp['growthRate'] = cols[2].number_input("Growth (%)", value=exp['growthRate'], key=f"ex_growth_{i}_{key_suffix}")
        if cols[3].button("🗑️", key=f"ex_del_{i}_{key_suffix}", help="Remove this expense item"):
            active_scenario['expenses'].pop(i)
            st.rerun()

    if st.button("Add Expense", key=f"add_expense_{key_suffix}"):
        active_scenario['expenses'].append({'name': 'Living Expenses', 'amount': 50000, 'growthRate': 3})
        st.rerun()

# --- Main App Layout ---
st.set_page_config(layout="wide")
st.title("📈 Integrated Retirement & Tax Planner")
st.markdown("An advanced simulator combining long-term retirement planning with detailed annual tax calculations.")

# --- Initialize Session State ---
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = [{
        'name': 'My Retirement Plan', 'initialInvestment': 500000, 'investmentReturn': 6, 
        'birthYear': 1980, 'endYear': TODAY_YEAR + 40, 'us_dividend_account': 'Non-Registered',
        'incomes': [{'name': 'CPP/OAS', 'amount': 15000, 'startYear': TODAY_YEAR + 25, 'growthRate': 2.5, 'type': 'cpp_oas'}],
        'expenses': [{'name': 'Base Expenses', 'amount': 60000, 'growthRate': 3.0}]
    }]
if 'active_scenario_index' not in st.session_state:
    st.session_state.active_scenario_index = 0
if 'results' not in st.session_state:
    st.session_state.results = None

# --- UI LAYOUT RESTRUCTURED ---
with st.expander("⚙️ Settings & Inputs", expanded=True):
    # --- Scenario Manager ---
    st.markdown("<h5>Scenario Manager</h5>", unsafe_allow_html=True)
    
    sc_cols = st.columns([2,1,1,1])
    scenario_names = [s['name'] for s in st.session_state.scenarios]
    st.session_state.active_scenario_index = scenario_names.index(sc_cols[0].selectbox("Active Scenario", scenario_names, index=st.session_state.active_scenario_index))
    
    if sc_cols[1].button("Add New", use_container_width=True, disabled=len(st.session_state.scenarios) >= 3):
        new_scenario = {'name': f'Scenario {len(st.session_state.scenarios) + 1}', 'initialInvestment': 500000, 'investmentReturn': 6, 'birthYear': 1980, 'endYear': TODAY_YEAR + 40, 'us_dividend_account': 'Non-Registered', 'incomes': [], 'expenses': []}
        st.session_state.scenarios.append(new_scenario)
        st.session_state.active_scenario_index = len(st.session_state.scenarios) - 1
        st.rerun()

    if sc_cols[2].button("Copy", use_container_width=True, disabled=len(st.session_state.scenarios) >= 3):
        scenario_to_copy = copy.deepcopy(st.session_state.scenarios[st.session_state.active_scenario_index])
        scenario_to_copy['name'] = f"{scenario_to_copy['name']} (Copy)"
        st.session_state.scenarios.append(scenario_to_copy)
        st.session_state.active_scenario_index = len(st.session_state.scenarios) - 1
        st.rerun()

    if sc_cols[3].button("Delete", use_container_width=True, disabled=len(st.session_state.scenarios) <= 1):
        st.session_state.scenarios.pop(st.session_state.active_scenario_index)
        st.session_state.active_scenario_index = 0
        st.rerun()

    st.markdown("---")
    
    # --- General Settings ---
    st.markdown("<h5>General Settings</h5>", unsafe_allow_html=True)
    active_scenario = st.session_state.scenarios[st.session_state.active_scenario_index]
    active_scenario['name'] = st.text_input("Scenario Name", value=active_scenario['name'], key=f"name_{st.session_state.active_scenario_index}")
    
    cols = st.columns(4)
    active_scenario['initialInvestment'] = cols[0].number_input("Initial Investments ($)", value=active_scenario['initialInvestment'], format="%d", key=f"inv_{st.session_state.active_scenario_index}")
    active_scenario['investmentReturn'] = cols[1].number_input("Avg. Return (%)", value=active_scenario['investmentReturn'], key=f"ret_{st.session_state.active_scenario_index}")
    active_scenario['birthYear'] = cols[2].number_input("Birth Year", value=active_scenario['birthYear'], format="%d", key=f"birth_{st.session_state.active_scenario_index}")
    active_scenario['endYear'] = cols[3].number_input("End Year", value=active_scenario['endYear'], format="%d", key=f"end_{st.session_state.active_scenario_index}")
    
    st.markdown("<h5>Tax Settings</h5>", unsafe_allow_html=True)
    active_scenario['us_dividend_account'] = st.selectbox(
        "US Dividend Account Type", ("Non-Registered", "RRSP/RRIF", "TFSA"),
        index=["Non-Registered", "RRSP/RRIF", "TFSA"].index(active_scenario['us_dividend_account']),
        key=f"usd_acct_{st.session_state.active_scenario_index}",
        help="Select the account type where US dividends are received. This affects the tax calculation for all US dividend income items."
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)

    income_expense_cols = st.columns(2)
    with income_expense_cols[0]:
        income_input_ui(st.session_state.active_scenario_index)
    with income_expense_cols[1]:
        expense_input_ui(st.session_state.active_scenario_index)

# --- Simulation Runner ---
if st.button("🚀 Run & Compare All Scenarios", type="primary", use_container_width=True):
    exchange_rate = get_exchange_rate()
    with st.spinner("Calculating all scenarios..."):
        st.session_state.results = [run_simulation(s, exchange_rate) for s in st.session_state.scenarios]

# --- Results Display ---
st.header("📊 Simulation Results")
if st.session_state.results:
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    summary_data = []

    for i, result in enumerate(st.session_state.results):
        scenario = st.session_state.scenarios[i]
        if result and result['data']:
            years = [d['year'] for d in result['data']]
            balances = [d['balance'] for d in result['data']]
            ages = [d['age'] for d in result['data']]
            
            fig.add_trace(go.Scatter(
                x=years, y=balances, mode='lines+markers', name=scenario['name'],
                line=dict(color=colors[i % len(colors)], width=3), marker=dict(size=5),
                hovertext=[f"Age: {age}" for age in ages],
                hovertemplate='<b>%{data.name}</b><br><b>Year:</b> %{x}<br><b>Balance:</b> %{y:$,.0f}<br><b>%{hovertext}</b><extra></extra>'
            ))
            
            final_balance = balances[-1] if balances else 0
            depletion_text = f"{result['depletion_year']} (Age: {result['depletion_year'] - scenario['birthYear']})" if result['depletion_year'] else "Sustained"
            summary_data.append({
                "Scenario": scenario['name'], "Final Balance": format_currency(final_balance),
                "Funds Depleted In": depletion_text
            })

    fig.update_layout(
        title="Retirement Portfolio Projection",
        xaxis_title="Year", yaxis_title="Portfolio Balance",
        yaxis_tickprefix="$", yaxis_tickformat="~s",
        legend_title="Scenarios", template="plotly_dark", height=500,
        hovermode='x unified' # UI IMPROVEMENT: Vertical line hover
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h5>Results Summary</h5>", unsafe_allow_html=True)
    st.table(pd.DataFrame(summary_data).set_index("Scenario"))

    with st.expander("View Detailed Yearly Data"):
        selected_scenario_for_table = st.selectbox("Select scenario to view details", [s['name'] for s in st.session_state.scenarios])
        idx = [s['name'] for s in st.session_state.scenarios].index(selected_scenario_for_table)
        if st.session_state.results[idx]:
            df = pd.DataFrame(st.session_state.results[idx]['data'])
            df['balance'] = df['balance'].apply(format_currency)
            st.dataframe(df.set_index('year'), use_container_width=True)

else:
    st.info("Adjust settings in the expander above and click 'Run & Compare All Scenarios'.")
