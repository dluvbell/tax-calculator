import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime
import copy
import numpy as np

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
INCOME_TYPES = ['CPP', 'OAS', 'Pension/RRIF', 'Eligible Canadian Dividends', 'US Dividends', 'Interest', 'Capital Gains', 'Other Income']
INCOME_TYPE_MAP = {
    'CPP': 'cpp', 'OAS': 'oas', 'Pension/RRIF': 'pension_rrif', 'Eligible Canadian Dividends': 'cdn_dividends',
    'US Dividends': 'us_dividends', 'Interest': 'interest', 'Capital Gains': 'capital_gains', 'Other Income': 'other_income'
}

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data['rates']['CAD']
    except requests.exceptions.RequestException:
        return 1.35

def format_currency(amount):
    return f"${amount:,.0f}"

# --- Core Tax Calculation Logic ---
def calculate_progressive_tax(income, brackets):
    tax = 0
    previous_bracket_limit = 0
    for limit, rate in brackets.items():
        if income > limit:
            tax += (limit - previous_bracket_limit) * rate
            previous_bracket_limit = limit
        else:
            tax += (income - previous_bracket_limit) * rate
            break
    return tax

def calculate_after_tax_income(yearly_income_details, us_dividend_account):
    gross_income = sum(yearly_income_details.values())
    if gross_income == 0:
        return 0, 0

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

    total_taxable_income = sum([
        yearly_income_details.get(v, 0) for k, v in INCOME_TYPE_MAP.items() 
        if k not in ['Eligible Canadian Dividends', 'US Dividends', 'Capital Gains']
    ]) + grossed_up_dividends + taxable_us_dividends + capital_gains_taxable

    if total_taxable_income <= 0:
        return gross_income, gross_income

    federal_tax_before_credits = calculate_progressive_tax(total_taxable_income, FED_BRACKETS_2025)
    provincial_tax_before_credits = calculate_progressive_tax(total_taxable_income, ON_BRACKETS_2025)
    
    federal_tax, provincial_tax = federal_tax_before_credits, provincial_tax_before_credits

    federal_tax -= min(total_taxable_income, FED_BPA_2025) * list(FED_BRACKETS_2025.values())[0]
    provincial_tax -= min(total_taxable_income, ON_BPA_2025) * list(ON_BRACKETS_2025.values())[0]

    federal_tax -= grossed_up_dividends * FED_ELIGIBLE_DIVIDEND_CREDIT_RATE
    provincial_tax -= grossed_up_dividends * ON_ELIGIBLE_DIVIDEND_CREDIT_RATE

    if us_dividend_account == 'Non-Registered' and total_taxable_income > 0:
        canadian_tax_on_us_income = taxable_us_dividends * (federal_tax_before_credits / total_taxable_income)
        federal_tax -= min(us_withholding_tax, canadian_tax_on_us_income)

    federal_tax, provincial_tax = max(0, federal_tax), max(0, provincial_tax)

    net_income_for_clawback = total_taxable_income - (grossed_up_dividends - eligible_dividends_actual)
    oas_income = yearly_income_details.get('oas', 0) 
    oas_clawback = 0
    if net_income_for_clawback > OAS_CLAWBACK_THRESHOLD_2025:
        oas_clawback = (net_income_for_clawback - OAS_CLAWBACK_THRESHOLD_2025) * OAS_CLAWBACK_RATE
        oas_clawback = min(oas_clawback, oas_income)

    total_tax = federal_tax + provincial_tax + oas_clawback
    net_income = gross_income - total_tax
    return gross_income, net_income

# --- Core Retirement Simulation Engine ---
def run_simulation(scenario, exchange_rate, investment_return=None):
    errors = []
    if scenario['startYear'] > scenario['endYear']:
        errors.append("Retirement Start Year cannot be after End Year.")
    if scenario['birthYear'] >= scenario['startYear']:
        errors.append("Birth Year must be before Retirement Start Year.")
    if errors:
        return {'errors': errors}

    investment = float(scenario['initialInvestment'])
    
    # For deterministic simulation, use the value from the scenario
    if investment_return is None:
        investment_return = float(scenario['investmentReturn']) / 100

    years_to_grow = scenario['startYear'] - TODAY_YEAR
    if years_to_grow > 0:
        # For Monte Carlo, apply random returns only during retirement years
        if isinstance(investment_return, dict):
            pre_retirement_return = np.mean(list(investment_return.values()))
            investment *= (1 + pre_retirement_return) ** years_to_grow
        else:
            investment *= (1 + investment_return) ** years_to_grow

    yearly_data, depletion_year = [], None
    
    for year in range(scenario['startYear'], scenario['endYear'] + 1):
        gross_annual_income = 0
        net_annual_income = 0
        
        if investment <= 0 and depletion_year is None: depletion_year = year

        current_investment_return = investment_return
        if isinstance(investment_return, dict): # For Monte Carlo
            current_investment_return = investment_return.get(year, 0)

        for crash in scenario.get('marketCrashes', []):
            duration = int(crash.get('duration', 1))
            if duration > 0 and crash.get('timing') == 'start' and year >= int(crash['startYear']) and year < int(crash['startYear']) + duration:
                investment *= (1 - float(crash['totalDecline']) / 100) ** (1 / duration)

        for event in scenario.get('oneTimeEvents', []):
            if int(event['year']) == year:
                investment += float(event['amount']) if event['type'] == 'Income' else -float(event['amount'])

        yearly_income_details = {}
        for income in scenario['incomes']:
            if year >= int(income['startYear']):
                years_from_today = int(income['startYear']) - TODAY_YEAR
                amount_at_start = float(income['amount']) * ((1 + float(income['growthRate']) / 100) ** (years_from_today if years_from_today > 0 else 0))
                future_value = amount_at_start * ((1 + float(income['growthRate']) / 100) ** (year - int(income['startYear'])))
                
                income_type_key = INCOME_TYPE_MAP[income['type']]
                if income_type_key == 'us_dividends': future_value *= exchange_rate
                yearly_income_details[income_type_key] = yearly_income_details.get(income_type_key, 0) + future_value
        
        gross_annual_income, net_annual_income = calculate_after_tax_income(yearly_income_details, scenario['us_dividend_account'])
        net_annual_expense = sum(float(exp['amount']) * ((1 + float(exp['growthRate']) / 100) ** (year - TODAY_YEAR)) for exp in scenario['expenses'])
        
        investment += (net_annual_income - net_annual_expense)
        if investment > 0: investment *= (1 + current_investment_return)

        for crash in scenario.get('marketCrashes', []):
            duration = int(crash.get('duration', 1))
            if duration > 0 and crash.get('timing') == 'end' and year >= int(crash['startYear']) and year < int(crash['startYear']) + duration:
                investment *= (1 - float(crash['totalDecline']) / 100) ** (1 / duration)
        
        yearly_data.append({
            'year': year, 
            'age': year - scenario['birthYear'], 
            'gross_income': gross_annual_income,
            'net_income': net_annual_income,
            'balance': round(max(0, investment))
        })
            
    return {'data': yearly_data, 'depletion_year': depletion_year, 'errors': None}

# --- Monte Carlo Simulation ---
@st.cache_data(ttl=3600)
def run_monte_carlo_simulation(scenario, exchange_rate, mean_return, std_dev, num_simulations):
    all_sim_results = []
    num_success = 0
    
    for _ in range(num_simulations):
        sim_years = range(scenario['startYear'], scenario['endYear'] + 1)
        random_returns = np.random.normal(mean_return / 100, std_dev / 100, len(sim_years))
        yearly_returns = {year: ret for year, ret in zip(sim_years, random_returns)}
        
        result = run_simulation(scenario, exchange_rate, investment_return=yearly_returns)
        
        if result and not result['errors']:
            final_balance = result['data'][-1]['balance']
            if final_balance > 0:
                num_success += 1
            all_sim_results.append([d['balance'] for d in result['data']])

    if not all_sim_results:
        return None

    success_rate = (num_success / num_simulations) * 100
    results_by_year = np.array(all_sim_results).T
    
    percentiles = {
        'p10': [np.percentile(year_data, 10) for year_data in results_by_year],
        'p25': [np.percentile(year_data, 25) for year_data in results_by_year],
        'p50': [np.percentile(year_data, 50) for year_data in results_by_year],
        'p75': [np.percentile(year_data, 75) for year_data in results_by_year],
        'p90': [np.percentile(year_data, 90) for year_data in results_by_year],
    }
    
    return {
        'success_rate': success_rate,
        'percentiles': percentiles,
        'years': list(sim_years)
    }

# --- UI Components & Callbacks ---
def add_item(list_name, default_item):
    st.session_state.scenarios[st.session_state.active_scenario_index][list_name].append(default_item)

def delete_item(list_name, index):
    st.session_state.scenarios[st.session_state.active_scenario_index][list_name].pop(index)

def create_dynamic_list_ui(list_name, fields, title, default_item, scenario_index, birth_year):
    st.markdown(f"<h5>{title}</h5>", unsafe_allow_html=True)
    active_scenario = st.session_state.scenarios[scenario_index]
    if list_name not in active_scenario: active_scenario[list_name] = []

    for i, item in enumerate(active_scenario[list_name]):
        cols = st.columns([f['width'] for f in fields] + [1])
        for j, field in enumerate(fields):
            unique_key = f"scen_{scenario_index}_{list_name}_{i}_{field['key']}"
            if field['type'] == 'text': 
                item[field['key']] = cols[j].text_input(field['label'], value=item[field['key']], key=unique_key)
            elif field['type'] == 'number': 
                year_value = cols[j].number_input(field['label'], value=item[field['key']], key=unique_key)
                item[field['key']] = year_value
                if field['key'] in ['startYear', 'year']:
                    age = year_value - birth_year
                    cols[j].caption(f"Age: {age}")
            elif field['type'] == 'select': 
                item[field['key']] = cols[j].selectbox(field['label'], field['options'], index=field['options'].index(item[field['key']]), key=unique_key)
        
        cols[-1].markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
        delete_key = f"scen_{scenario_index}_{list_name}_del_{i}"
        cols[-1].button("🗑️", key=delete_key, help=f"Remove this item", on_click=delete_item, args=(list_name, i))

    add_key = f"scen_{scenario_index}_add_{list_name}"
    st.button(f"Add {title.replace('Recurring ','').replace('s','')}", key=add_key, use_container_width=True, on_click=add_item, args=(list_name, default_item))

# --- Main App ---
st.set_page_config(layout="wide")
st.markdown("""
<style>
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button {
        -webkit-appearance: none !important; margin: 0 !important;
    }
    input[type=number] { -moz-appearance: textfield !important; }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) > div[data-testid="stHorizontalBlock"] {
        display: flex; justify-content: flex-start; gap: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Integrated Retirement & Tax Planner")
st.markdown("An advanced simulator combining long-term retirement planning with detailed annual tax calculations.")

if 'scenarios' not in st.session_state:
    st.session_state.scenarios = [{
        'name': 'My Retirement Plan', 'initialInvestment': 500000, 'investmentReturn': 6, 'birthYear': 1980, 
        'startYear': TODAY_YEAR + 20, 'endYear': TODAY_YEAR + 50, 'us_dividend_account': 'Non-Registered', 
        'incomes': [
            {'type': 'OAS', 'amount': 8000, 'startYear': TODAY_YEAR + 25, 'growthRate': 2.5},
            {'type': 'CPP', 'amount': 9000, 'startYear': TODAY_YEAR + 25, 'growthRate': 2.5}
        ], 
        'expenses': [{'name': 'Base Expenses', 'amount': 60000, 'growthRate': 3.0}], 
        'oneTimeEvents': [], 'marketCrashes': []
    }]
if 'active_scenario_index' not in st.session_state: st.session_state.active_scenario_index = 0
if 'results' not in st.session_state: st.session_state.results = None
if 'mc_results' not in st.session_state: st.session_state.mc_results = None

with st.expander("⚙️ Settings & Inputs", expanded=True):
    # --- Scenario Manager ---
    def update_active_index(): 
        st.session_state.active_scenario_index = st.session_state.scenario_selector
        st.session_state.mc_results = None 
    def add_scenario_cb():
        new_scenario = {'name': f'Scenario {len(st.session_state.scenarios) + 1}', 'initialInvestment': 500000, 'investmentReturn': 6, 'birthYear': 1980, 'startYear': TODAY_YEAR + 20, 'endYear': TODAY_YEAR + 50, 'us_dividend_account': 'Non-Registered', 'incomes': [], 'expenses': [], 'oneTimeEvents': [], 'marketCrashes': []}
        st.session_state.scenarios.append(new_scenario)
        st.session_state.active_scenario_index = len(st.session_state.scenarios) - 1
    def copy_scenario_cb():
        scenario_to_copy = copy.deepcopy(st.session_state.scenarios[st.session_state.active_scenario_index])
        scenario_to_copy['name'] = f"{scenario_to_copy['name']} (Copy)"
        st.session_state.scenarios.append(scenario_to_copy)
        st.session_state.active_scenario_index = len(st.session_state.scenarios) - 1
    
    def delete_scenario_cb():
        index_to_delete = st.session_state.active_scenario_index
        st.session_state.scenarios.pop(index_to_delete)
        if st.session_state.results and len(st.session_state.results) > index_to_delete:
            st.session_state.results.pop(index_to_delete)
        st.session_state.active_scenario_index = 0
        st.session_state.mc_results = None

    st.markdown("<h5>Scenario Manager</h5>", unsafe_allow_html=True)
    left_col, right_col = st.columns([2, 1])
    with left_col:
        scenario_names = [s['name'] for s in st.session_state.scenarios]
        st.selectbox("Active Scenario", options=range(len(scenario_names)), format_func=lambda x: scenario_names[x] if x < len(scenario_names) else "", index=st.session_state.active_scenario_index, key="scenario_selector", on_change=update_active_index, label_visibility="collapsed")
    with right_col:
        b1, b2, b3 = st.columns(3)
        b1.button("➕", help="Add a new scenario", disabled=len(st.session_state.scenarios) >= 5, on_click=add_scenario_cb, use_container_width=True)
        b2.button("📋", help="Copy the current scenario", disabled=len(st.session_state.scenarios) >= 5, on_click=copy_scenario_cb, use_container_width=True)
        b3.button("🗑️", help="Delete the current scenario", disabled=len(st.session_state.scenarios) <= 1, on_click=delete_scenario_cb, use_container_width=True)
    
    st.markdown("---")
    
    active_scenario_index = st.session_state.active_scenario_index
    if not st.session_state.scenarios:
        st.warning("All scenarios have been deleted. Please add a new one.")
    else:
        active_scenario = st.session_state.scenarios[active_scenario_index]
        edit_section = st.selectbox("Edit Section", ["General & Tax Settings", "Recurring Incomes", "Recurring Expenses", "One-Time Events", "Market Volatility"], key=f"scen_{active_scenario_index}_edit_section")
        
        if edit_section == "General & Tax Settings":
            active_scenario['name'] = st.text_input("Scenario Name", value=active_scenario['name'], key=f"scen_{active_scenario_index}_name")
            cols = st.columns(5)
            active_scenario['initialInvestment'] = cols[0].number_input("Initial Inv. ($)", value=active_scenario['initialInvestment'], format="%d", key=f"scen_{active_scenario_index}_inv")
            active_scenario['investmentReturn'] = cols[1].number_input("Avg. Return (%)", value=active_scenario['investmentReturn'], key=f"scen_{active_scenario_index}_ret")
            birth_year_val = cols[2].number_input("Birth Year", value=active_scenario['birthYear'], format="%d", key=f"scen_{active_scenario_index}_birth")
            active_scenario['birthYear'] = birth_year_val
            start_year_val = cols[3].number_input("Retirement Start Year", value=active_scenario['startYear'], format="%d", key=f"scen_{active_scenario_index}_start")
            active_scenario['startYear'] = start_year_val
            cols[3].caption(f"Age: {start_year_val - birth_year_val}")
            end_year_val = cols[4].number_input("End Year", value=active_scenario['endYear'], format="%d", key=f"scen_{active_scenario_index}_end")
            active_scenario['endYear'] = end_year_val
            cols[4].caption(f"Age: {end_year_val - birth_year_val}")
            st.markdown("<h6>Tax Settings</h6>", unsafe_allow_html=True)
            active_scenario['us_dividend_account'] = st.selectbox("US Dividend Account Type", ("Non-Registered", "RRSP/RRIF", "TFSA"), index=["Non-Registered", "RRSP/RRIF", "TFSA"].index(active_scenario['us_dividend_account']), key=f"scen_{active_scenario_index}_usdiv")
        
        else:
            birth_year = active_scenario.get('birthYear', TODAY_YEAR - 40)
            if edit_section == "Recurring Incomes":
                create_dynamic_list_ui('incomes', [{'key': 'type', 'label': 'Type', 'type': 'select', 'options': INCOME_TYPES, 'default': 'Other Income', 'width': 3}, {'key': 'amount', 'label': "Amount", 'type': 'number', 'default': 10000, 'width': 2}, {'key': 'startYear', 'label': 'Start', 'type': 'number', 'default': TODAY_YEAR + 10, 'width': 2}, {'key': 'growthRate', 'label': 'Growth', 'type': 'number', 'default': 2.5, 'width': 2}], 'Recurring Incomes', {'type': 'Other Income', 'amount': 10000, 'startYear': TODAY_YEAR + 10, 'growthRate': 2.5}, active_scenario_index, birth_year)
            elif edit_section == "Recurring Expenses":
                create_dynamic_list_ui('expenses', [{'key': 'name', 'label': 'Name', 'type': 'text', 'default': 'Living Expenses', 'width': 5}, {'key': 'amount', 'label': "Amount", 'type': 'number', 'default': 50000, 'width': 3}, {'key': 'growthRate', 'label': 'Growth', 'type': 'number', 'default': 3, 'width': 3}], 'Recurring Expenses', {'name': 'Living Expenses', 'amount': 50000, 'growthRate': 3}, active_scenario_index, birth_year)
            elif edit_section == "One-Time Events":
                create_dynamic_list_ui('oneTimeEvents', [{'key': 'name', 'label': 'Event Name', 'type': 'text', 'default': 'New Event', 'width': 4}, {'key': 'type', 'label': 'Type', 'type': 'select', 'options': ['Income', 'Expense'], 'default': 'Expense', 'width': 2}, {'key': 'amount', 'label': 'Amount ($)', 'type': 'number', 'default': 20000, 'width': 2}, {'key': 'year', 'label': 'Year', 'type': 'number', 'default': TODAY_YEAR + 15, 'width': 2}], 'One-Time Events', {'name': 'New Event', 'type': 'Expense', 'amount': 20000, 'year': TODAY_YEAR + 15}, active_scenario_index, birth_year)
            elif edit_section == "Market Volatility":
                create_dynamic_list_ui('marketCrashes', [{'key': 'startYear', 'label': 'Crash Start', 'type': 'number', 'default': TODAY_YEAR + 10, 'width': 2}, {'key': 'duration', 'label': 'Duration', 'type': 'number', 'default': 2, 'width': 2}, {'key': 'totalDecline', 'label': 'Decline (%)', 'type': 'number', 'default': 30, 'width': 2}, {'key': 'timing', 'label': 'Timing', 'type': 'select', 'options': ['start', 'end'], 'default': 'start', 'width': 2}], 'Market Volatility', {'startYear': TODAY_YEAR + 10, 'duration': 2, 'totalDecline': 30, 'timing': 'start'}, active_scenario_index, birth_year)

# --- Deterministic Simulation Runner ---
if st.button("🚀 Run & Compare All Scenarios", type="primary", use_container_width=True):
    exchange_rate = get_exchange_rate()
    with st.spinner("Calculating all scenarios..."):
        st.session_state.results = [run_simulation(s, exchange_rate) for s in st.session_state.scenarios]
    st.session_state.mc_results = None 

# --- Results Display ---
st.header("📊 Simulation Results")
if st.session_state.results:
    if len(st.session_state.results) > len(st.session_state.scenarios):
        st.session_state.results = st.session_state.results[:len(st.session_state.scenarios)]

    has_errors = any(res.get('errors') for res in st.session_state.results)
    if has_errors:
        for i, result in enumerate(st.session_state.results):
            if i < len(st.session_state.scenarios) and result.get('errors'):
                st.error(f"**Scenario '{st.session_state.scenarios[i]['name']}' has input errors:**")
                for error in result['errors']:
                    st.warning(f"- {error}")
    else:
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        symbols = ['circle', 'square', 'diamond', 'cross', 'x']
        summary_data = []

        for i, result in enumerate(st.session_state.results):
            if i < len(st.session_state.scenarios):
                scenario = st.session_state.scenarios[i]
                if result and result.get('data'):
                    years = [d['year'] for d in result['data']]
                    balances = [d['balance'] for d in result['data']]
                    ages = [d['age'] for d in result['data']]
                    
                    fig.add_trace(go.Scatter(x=years, y=balances, mode='lines+markers', name=scenario['name'], line=dict(color=colors[i % len(colors)], width=3), marker=dict(size=7, symbol=symbols[i % len(symbols)]), hovertext=[f"Age: {age}" for age in ages], hovertemplate='<b>%{data.name}</b><br><b>Year:</b> %{x}<br><b>Balance:</b> %{y:$,.0f}<br><b>%{hovertext}</b><extra></extra>'))
                    
                    final_balance = balances[-1] if balances else 0
                    depletion_text = f"{result['depletion_year']} (Age: {result['depletion_year'] - scenario['birthYear']})" if result['depletion_year'] else "Sustained"
                    summary_data.append({"Scenario": scenario['name'], "Final Balance": format_currency(final_balance), "Funds Depleted In": depletion_text})

        fig.update_layout(title="Retirement Portfolio Projection", xaxis_title="Year", yaxis_title="Portfolio Balance", yaxis_tickprefix="$", yaxis_tickformat="~s", legend_title="Scenarios", template="plotly_dark", height=500, hovermode='x unified', xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig, use_container_width=True)

        if summary_data:
            st.markdown("<h5>Results Summary</h5>", unsafe_allow_html=True)
            st.table(pd.DataFrame(summary_data).set_index("Scenario"))

        with st.expander("View Detailed Yearly Data"):
            scenario_names_for_details = [s['name'] for s in st.session_state.scenarios]
            if scenario_names_for_details:
                selected_scenario_for_table = st.selectbox("Select scenario to view details", scenario_names_for_details)
                idx = scenario_names_for_details.index(selected_scenario_for_table)
                if st.session_state.results and len(st.session_state.results) > idx and st.session_state.results[idx].get('data'):
                    df = pd.DataFrame(st.session_state.results[idx]['data'])
                    df['gross_income'] = df['gross_income'].apply(format_currency)
                    df['net_income'] = df['net_income'].apply(format_currency)
                    df['balance'] = df['balance'].apply(format_currency)
                    df_display = df[['year', 'age', 'gross_income', 'net_income', 'balance']]
                    st.dataframe(df_display.set_index('year'), use_container_width=True)

elif not st.session_state.scenarios:
     st.info("Please add a new scenario to begin.")
else:
    st.info("Adjust settings in the expander above and click 'Run & Compare All Scenarios'.")

# --- Monte Carlo Simulation Section ---
with st.expander("🔬 Monte Carlo Simulation"):
    if not st.session_state.scenarios:
        st.warning("Please add a scenario first.")
    else:
        active_scenario = st.session_state.scenarios[st.session_state.active_scenario_index]
        st.info(f"This analysis will run on your currently selected scenario: **{active_scenario['name']}**")

        mc_cols = st.columns(3)
        mean_return = mc_cols[0].number_input("Mean Annual Return (%)", value=6.0, help="The average long-term expected return of your portfolio.")
        std_dev = mc_cols[1].number_input("Annual Volatility (Std. Dev %)", value=12.0, help="The typical range of portfolio fluctuation. A higher value means more risk. (e.g., 60/40 portfolio ~12%)")
        num_simulations = mc_cols[2].slider("Number of Simulations", min_value=100, max_value=1000, value=500, step=100, help="More simulations provide a more accurate probability but take longer to run.")

        if st.button("🎲 Run Monte Carlo"):
            exchange_rate = get_exchange_rate()
            with st.spinner(f"Running {num_simulations} simulations..."):
                st.session_state.mc_results = run_monte_carlo_simulation(active_scenario, exchange_rate, mean_return, std_dev, num_simulations)
        
        if st.session_state.mc_results:
            results = st.session_state.mc_results
            st.subheader("Monte Carlo Results")
            
            st.metric(label="Retirement Plan Success Rate", value=f"{results['success_rate']:.1f}%")
            
            fig = go.Figure()
            years = results['years']
            p = results['percentiles']
            
            fig.add_trace(go.Scatter(x=years, y=p['p90'], mode='lines', line=dict(width=0), name='90th Percentile', showlegend=False))
            fig.add_trace(go.Scatter(x=years, y=p['p10'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 80, 0.2)', name='10th-90th Percentile', showlegend=True))
            
            fig.add_trace(go.Scatter(x=years, y=p['p75'], mode='lines', line=dict(width=0), name='75th Percentile', showlegend=False))
            fig.add_trace(go.Scatter(x=years, y=p['p25'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 176, 246, 0.2)', name='25th-75th Percentile', showlegend=True))

            fig.add_trace(go.Scatter(x=years, y=p['p50'], mode='lines', line=dict(color='white', width=3), name='Median Outcome'))

            fig.update_layout(title="Monte Carlo Portfolio Projection", xaxis_title="Year", yaxis_title="Portfolio Balance", yaxis_tickprefix="$", yaxis_tickformat="~s", legend_title="Outcome Percentiles", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### Final Balance Summary")
            summary_cols = st.columns(3)
            summary_cols[0].metric("Worst 10% Outcome", format_currency(p['p10'][-1]))
            summary_cols[1].metric("Median Outcome", format_currency(p['p50'][-1]))
            summary_cols[2].metric("Best 10% Outcome", format_currency(p['p90'][-1]))
