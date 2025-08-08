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
    # Pre-retirement growth
    investment = float(scenario['initialInvestment'])
    years_to_grow = scenario['startYear'] - TODAY_YEAR
    if years_to_grow > 0:
        investment *= (1 + float(scenario['investmentReturn']) / 100) ** years_to_grow

    yearly_data = []
    depletion_year = None

    for year in range(scenario['startYear'], scenario['endYear'] + 1):
        if investment <= 0 and depletion_year is None:
            depletion_year = year
        
        balance = max(0, investment)
        age = year - scenario['birthYear']
        yearly_data.append({'year': year, 'age': age, 'balance': round(balance)})

        # Apply start-of-year market crashes
        for crash in scenario.get('marketCrashes', []):
            duration = int(crash.get('duration', 1))
            if duration > 0 and crash.get('timing') == 'start' and year >= int(crash['startYear']) and year < int(crash['startYear']) + duration:
                decline_rate = (1 - float(crash['totalDecline']) / 100) ** (1 / duration)
                investment *= decline_rate

        # Apply one-time events
        for event in scenario.get('oneTimeEvents', []):
            if int(event['year']) == year:
                amount = float(event['amount'])
                investment += amount if event['type'] == 'Income' else -amount

        # Calculate recurring incomes/expenses
        yearly_income_details = {}
        for income in scenario['incomes']:
            if year >= int(income['startYear']):
                years_from_today = int(income['startYear']) - TODAY_YEAR
                amount_at_start = float(income['amount']) * ((1 + float(income['growthRate']) / 100) ** (years_from_today if years_from_today > 0 else 0))
                years_since_start = year - int(income['startYear'])
                future_value = amount_at_start * ((1 + float(income['growthRate']) / 100) ** years_since_start)
                income_type = income.get('type', 'other_income')
                yearly_income_details[income_type] = yearly_income_details.get(income_type, 0) + future_value
        
        net_annual_income = calculate_after_tax_income(yearly_income_details, scenario['us_dividend_account'])
        net_annual_expense = sum(float(exp['amount']) * ((1 + float(exp['growthRate']) / 100) ** (year - TODAY_YEAR)) for exp in scenario['expenses'])
        
        investment += (net_annual_income - net_annual_expense)
        
        if investment > 0:
            investment *= (1 + float(scenario['investmentReturn']) / 100)

        # Apply end-of-year market crashes
        for crash in scenario.get('marketCrashes', []):
            duration = int(crash.get('duration', 1))
            if duration > 0 and crash.get('timing') == 'end' and year >= int(crash['startYear']) and year < int(crash['startYear']) + duration:
                decline_rate = (1 - float(crash['totalDecline']) / 100) ** (1 / duration)
                investment *= decline_rate
            
    return {'data': yearly_data, 'depletion_year': depletion_year}

# --- UI Components ---
def create_dynamic_list_ui(list_name, fields, title, key_suffix):
    st.markdown(f"<h6>{title}</h6>", unsafe_allow_html=True)
    active_scenario = st.session_state.scenarios[st.session_state.active_scenario_index]
    if list_name not in active_scenario: active_scenario[list_name] = []

    for i, item in enumerate(active_scenario[list_name]):
        cols = st.columns([f['width'] for f in fields] + [1])
        for j, field in enumerate(fields):
            if field['type'] == 'text':
                item[field['key']] = cols[j].text_input(field['label'], value=item[field['key']], key=f"{list_name}_{i}_{field['key']}_{key_suffix}")
            elif field['type'] == 'number':
                item[field['key']] = cols[j].number_input(field['label'], value=item[field['key']], key=f"{list_name}_{i}_{field['key']}_{key_suffix}")
            elif field['type'] == 'select':
                item[field['key']] = cols[j].selectbox(field['label'], field['options'], index=field['options'].index(item[field['key']]), key=f"{list_name}_{i}_{field['key']}_{key_suffix}")
        
        if cols[-1].button("🗑️", key=f"{list_name}_del_{i}_{key_suffix}", help=f"Remove this item"):
            active_scenario[list_name].pop(i)
            st.rerun()

    if st.button(f"Add {title.replace('Recurring ','').replace('s','')}", key=f"add_{list_name}_{key_suffix}"):
        new_item = {f['key']: f['default'] for f in fields}
        active_scenario[list_name].append(new_item)
        st.rerun()

# --- Main App Layout ---
st.set_page_config(layout="wide")
st.title("📈 Integrated Retirement & Tax Planner")
st.markdown("An advanced simulator combining long-term retirement planning with detailed annual tax calculations.")

# --- Initialize Session State ---
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = [{
        'name': 'My Retirement Plan', 'initialInvestment': 500000, 'investmentReturn': 6, 
        'birthYear': 1980, 'startYear': TODAY_YEAR + 20, 'endYear': TODAY_YEAR + 50, 'us_dividend_account': 'Non-Registered',
        'incomes': [{'name': 'CPP/OAS', 'amount': 15000, 'startYear': TODAY_YEAR + 25, 'growthRate': 2.5, 'type': 'cpp_oas'}],
        'expenses': [{'name': 'Base Expenses', 'amount': 60000, 'growthRate': 3.0}],
        'oneTimeEvents': [], 'marketCrashes': []
    }]
if 'active_scenario_index' not in st.session_state:
    st.session_state.active_scenario_index = 0
if 'results' not in st.session_state:
    st.session_state.results = None

# --- UI LAYOUT RESTRUCTURED ---
with st.expander("⚙️ Settings & Inputs", expanded=True):
    st.markdown("<h5>Scenario Manager</h5>", unsafe_allow_html=True)
    sc_cols = st.columns([2,1,1,1])
    # ... (Scenario manager UI remains the same)
    scenario_names = [s['name'] for s in st.session_state.scenarios]
    st.session_state.active_scenario_index = scenario_names.index(sc_cols[0].selectbox("Active Scenario", scenario_names, index=st.session_state.active_scenario_index, key=f"selector_{st.session_state.active_scenario_index}"))
    
    if sc_cols[1].button("Add New", use_container_width=True, disabled=len(st.session_state.scenarios) >= 3):
        new_scenario = {'name': f'Scenario {len(st.session_state.scenarios) + 1}', 'initialInvestment': 500000, 'investmentReturn': 6, 'birthYear': 1980, 'startYear': TODAY_YEAR + 20, 'endYear': TODAY_YEAR + 50, 'us_dividend_account': 'Non-Registered', 'incomes': [], 'expenses': [], 'oneTimeEvents': [], 'marketCrashes': []}
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
    
    active_scenario = st.session_state.scenarios[st.session_state.active_scenario_index]
    key_suffix = st.session_state.active_scenario_index

    st.markdown("<h5>General Settings</h5>", unsafe_allow_html=True)
    active_scenario['name'] = st.text_input("Scenario Name", value=active_scenario['name'], key=f"name_{key_suffix}")
    cols = st.columns(5)
    active_scenario['initialInvestment'] = cols[0].number_input("Initial Investments ($)", value=active_scenario['initialInvestment'], format="%d", key=f"inv_{key_suffix}")
    active_scenario['investmentReturn'] = cols[1].number_input("Avg. Return (%)", value=active_scenario['investmentReturn'], key=f"ret_{key_suffix}")
    active_scenario['birthYear'] = cols[2].number_input("Birth Year", value=active_scenario['birthYear'], format="%d", key=f"birth_{key_suffix}")
    active_scenario['startYear'] = cols[3].number_input("Retirement Start Year", value=active_scenario['startYear'], format="%d", key=f"start_{key_suffix}")
    active_scenario['endYear'] = cols[4].number_input("End Year", value=active_scenario['endYear'], format="%d", key=f"end_{key_suffix}")
    
    st.markdown("<h5>Tax Settings</h5>", unsafe_allow_html=True)
    active_scenario['us_dividend_account'] = st.selectbox(
        "US Dividend Account Type", ("Non-Registered", "RRSP/RRIF", "TFSA"),
        index=["Non-Registered", "RRSP/RRIF", "TFSA"].index(active_scenario['us_dividend_account']), key=f"usd_acct_{key_suffix}"
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    income_expense_cols = st.columns(2)
    with income_expense_cols[0]:
        create_dynamic_list_ui('incomes', [{'key': 'name', 'label': 'Name', 'type': 'text', 'default': 'New Income', 'width': 3}, {'key': 'amount', 'label': "Amount (Today's $)", 'type': 'number', 'default': 10000, 'width': 2}, {'key': 'startYear', 'label': 'Start Year', 'type': 'number', 'default': TODAY_YEAR + 10, 'width': 2}, {'key': 'growthRate', 'label': 'Growth (%)', 'type': 'number', 'default': 2.5, 'width': 2}], 'Recurring Incomes', key_suffix)
    with income_expense_cols[1]:
        create_dynamic_list_ui('expenses', [{'key': 'name', 'label': 'Name', 'type': 'text', 'default': 'Living Expenses', 'width': 5}, {'key': 'amount', 'label': "Amount (Today's $)", 'type': 'number', 'default': 50000, 'width': 3}, {'key': 'growthRate', 'label': 'Growth (%)', 'type': 'number', 'default': 3, 'width': 3}], 'Recurring Expenses', key_suffix)

    with st.expander("One-Time Events (Inheritance, Car Purchase, etc.)"):
        create_dynamic_list_ui('oneTimeEvents', [{'key': 'name', 'label': 'Event Name', 'type': 'text', 'default': 'New Event', 'width': 4}, {'key': 'type', 'label': 'Type', 'type': 'select', 'options': ['Income', 'Expense'], 'default': 'Expense', 'width': 2}, {'key': 'amount', 'label': 'Amount ($)', 'type': 'number', 'default': 20000, 'width': 2}, {'key': 'year', 'label': 'Year', 'type': 'number', 'default': TODAY_YEAR + 15, 'width': 2}], 'One-Time Events', key_suffix)

    with st.expander("Market Volatility (Crashes)"):
        create_dynamic_list_ui('marketCrashes', [{'key': 'startYear', 'label': 'Crash Start Year', 'type': 'number', 'default': TODAY_YEAR + 10, 'width': 2}, {'key': 'duration', 'label': 'Duration (Yrs)', 'type': 'number', 'default': 2, 'width': 2}, {'key': 'totalDecline', 'label': 'Total Decline (%)', 'type': 'number', 'default': 30, 'width': 2}, {'key': 'timing', 'label': 'Timing', 'type': 'select', 'options': ['start', 'end'], 'default': 'start', 'width': 2}], 'Market Crashes', key_suffix)

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
        hovermode='x unified'
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
