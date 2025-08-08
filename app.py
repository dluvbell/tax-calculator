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
INCOME_TYPES = ['CPP/OAS', 'Pension/RRIF', 'Eligible Canadian Dividends', 'US Dividends', 'Interest', 'Capital Gains', 'Other Income']
INCOME_TYPE_MAP = {
    'CPP/OAS': 'cpp_oas', 'Pension/RRIF': 'pension_rrif', 'Eligible Canadian Dividends': 'cdn_dividends',
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

    if total_taxable_income <= 0: return gross_income

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
    oas_income = yearly_income_details.get('cpp_oas', 0) # Approximation
    oas_clawback = 0
    if net_income_for_clawback > OAS_CLAWBACK_THRESHOLD_2025:
        oas_clawback = (net_income_for_clawback - OAS_CLAWBACK_THRESHOLD_2025) * OAS_CLAWBACK_RATE
        oas_clawback = min(oas_clawback, oas_income)

    total_tax = federal_tax + provincial_tax + oas_clawback
    return gross_income - total_tax

# --- Core Retirement Simulation Engine ---
def run_simulation(scenario, exchange_rate):
    # Data Validation
    errors = []
    if scenario['startYear'] > scenario['endYear']:
        errors.append("Retirement Start Year cannot be after End Year.")
    if scenario['birthYear'] >= scenario['startYear']:
        errors.append("Birth Year must be before Retirement Start Year.")
    if errors:
        return {'errors': errors}

    investment = float(scenario['initialInvestment'])
    years_to_grow = scenario['startYear'] - TODAY_YEAR
    if years_to_grow > 0:
        investment *= (1 + float(scenario['investmentReturn']) / 100) ** years_to_grow

    yearly_data, depletion_year = [], None

    for year in range(scenario['startYear'], scenario['endYear'] + 1):
        if investment <= 0 and depletion_year is None: depletion_year = year
        yearly_data.append({'year': year, 'age': year - scenario['birthYear'], 'balance': round(max(0, investment))})

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
        
        net_annual_income = calculate_after_tax_income(yearly_income_details, scenario['us_dividend_account'])
        net_annual_expense = sum(float(exp['amount']) * ((1 + float(exp['growthRate']) / 100) ** (year - TODAY_YEAR)) for exp in scenario['expenses'])
        
        investment += (net_annual_income - net_annual_expense)
        if investment > 0: investment *= (1 + float(scenario['investmentReturn']) / 100)

        for crash in scenario.get('marketCrashes', []):
            duration = int(crash.get('duration', 1))
            if duration > 0 and crash.get('timing') == 'end' and year >= int(crash['startYear']) and year < int(crash['startYear']) + duration:
                investment *= (1 - float(crash['totalDecline']) / 100) ** (1 / duration)
            
    return {'data': yearly_data, 'depletion_year': depletion_year, 'errors': None}

# --- UI Components & Callbacks ---
def add_item(list_name, default_item):
    st.session_state.scenarios[st.session_state.active_scenario_index][list_name].append(default_item)

def delete_item(list_name, index):
    st.session_state.scenarios[st.session_state.active_scenario_index][list_name].pop(index)

def create_dynamic_list_ui(list_name, fields, title, default_item):
    st.markdown(f"<h5>{title}</h5>", unsafe_allow_html=True)
    active_scenario = st.session_state.scenarios[st.session_state.active_scenario_index]
    if list_name not in active_scenario: active_scenario[list_name] = []

    for i, item in enumerate(active_scenario[list_name]):
        cols = st.columns([f['width'] for f in fields] + [1])
        for j, field in enumerate(fields):
            if field['type'] == 'text': item[field['key']] = cols[j].text_input(field['label'], value=item[field['key']], key=f"{list_name}_{i}_{field['key']}")
            elif field['type'] == 'number': item[field['key']] = cols[j].number_input(field['label'], value=item[field['key']], key=f"{list_name}_{i}_{field['key']}")
            elif field['type'] == 'select': item[field['key']] = cols[j].selectbox(field['label'], field['options'], index=field['options'].index(item[field['key']]), key=f"{list_name}_{i}_{field['key']}")
        
        cols[-1].markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
        cols[-1].button("🗑️", key=f"{list_name}_del_{i}", help=f"Remove this item", on_click=delete_item, args=(list_name, i))

    st.button(f"Add {title.replace('Recurring ','').replace('s','')}", key=f"add_{list_name}", use_container_width=True, on_click=add_item, args=(list_name, default_item))

# --- Main App ---
st.set_page_config(layout="wide")
# CSS to hide number input steppers. Using !important to ensure it overrides default styles.
st.markdown("""<style>
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
    }
    input[type=number] {
        -moz-appearance: textfield !important;
    }
</style>""", unsafe_allow_html=True)

st.title("📈 Integrated Retirement & Tax Planner")
st.markdown("An advanced simulator combining long-term retirement planning with detailed annual tax calculations.")

if 'scenarios' not in st.session_state:
    st.session_state.scenarios = [{'name': 'My Retirement Plan', 'initialInvestment': 500000, 'investmentReturn': 6, 'birthYear': 1980, 'startYear': TODAY_YEAR + 20, 'endYear': TODAY_YEAR + 50, 'us_dividend_account': 'Non-Registered', 'incomes': [{'type': 'CPP/OAS', 'amount': 15000, 'startYear': TODAY_YEAR + 25, 'growthRate': 2.5}], 'expenses': [{'name': 'Base Expenses', 'amount': 60000, 'growthRate': 3.0}], 'oneTimeEvents': [], 'marketCrashes': []}]
if 'active_scenario_index' not in st.session_state: st.session_state.active_scenario_index = 0
if 'results' not in st.session_state: st.session_state.results = None

with st.expander("⚙️ Settings & Inputs", expanded=True):
    # --- Scenario Manager ---
    def update_active_index(): st.session_state.active_scenario_index = st.session_state.scenario_selector
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
        st.session_state.scenarios.pop(st.session_state.active_scenario_index)
        st.session_state.active_scenario_index = 0

    st.markdown("<h5>Scenario Manager</h5>", unsafe_allow_html=True)
    scenario_names = [s['name'] for s in st.session_state.scenarios]
    st.selectbox("Active Scenario", options=range(len(scenario_names)), format_func=lambda x: scenario_names[x], index=st.session_state.active_scenario_index, key="scenario_selector", on_change=update_active_index)
    
    sc_cols = st.columns(3)
    sc_cols[0].button("➕", use_container_width=True, help="Add a new scenario", disabled=len(st.session_state.scenarios) >= 5, on_click=add_scenario_cb)
    sc_cols[1].button("📋", use_container_width=True, help="Copy the current scenario", disabled=len(st.session_state.scenarios) >= 5, on_click=copy_scenario_cb)
    sc_cols[2].button("🗑️", use_container_width=True, help="Delete the current scenario", disabled=len(st.session_state.scenarios) <= 1, on_click=delete_scenario_cb)
    
    st.markdown("---")
    
    active_scenario = st.session_state.scenarios[st.session_state.active_scenario_index]
    edit_section = st.selectbox("Edit Section", ["General & Tax Settings", "Recurring Incomes", "Recurring Expenses", "One-Time Events", "Market Volatility"])
    
    if edit_section == "General & Tax Settings":
        active_scenario['name'] = st.text_input("Scenario Name", value=active_scenario['name'])
        cols = st.columns(5)
        active_scenario['initialInvestment'] = cols[0].number_input("Initial Inv. ($)", value=active_scenario['initialInvestment'], format="%d")
        active_scenario['investmentReturn'] = cols[1].number_input("Avg. Return (%)", value=active_scenario['investmentReturn'])
        active_scenario['birthYear'] = cols[2].number_input("Birth Year", value=active_scenario['birthYear'], format="%d")
        active_scenario['startYear'] = cols[3].number_input("Retirement Start Year", value=active_scenario['startYear'], format="%d")
        active_scenario['endYear'] = cols[4].number_input("End Year", value=active_scenario['endYear'], format="%d")
        st.markdown("<h6>Tax Settings</h6>", unsafe_allow_html=True)
        active_scenario['us_dividend_account'] = st.selectbox("US Dividend Account Type", ("Non-Registered", "RRSP/RRIF", "TFSA"), index=["Non-Registered", "RRSP/RRIF", "TFSA"].index(active_scenario['us_dividend_account']))
    
    elif edit_section == "Recurring Incomes":
        create_dynamic_list_ui('incomes', [{'key': 'type', 'label': 'Type', 'type': 'select', 'options': INCOME_TYPES, 'default': 'Other Income', 'width': 3}, {'key': 'amount', 'label': "Amount", 'type': 'number', 'default': 10000, 'width': 2}, {'key': 'startYear', 'label': 'Start', 'type': 'number', 'default': TODAY_YEAR + 10, 'width': 2}, {'key': 'growthRate', 'label': 'Growth', 'type': 'number', 'default': 2.5, 'width': 2}], 'Recurring Incomes', {'type': 'Other Income', 'amount': 10000, 'startYear': TODAY_YEAR + 10, 'growthRate': 2.5})
    elif edit_section == "Recurring Expenses":
        create_dynamic_list_ui('expenses', [{'key': 'name', 'label': 'Name', 'type': 'text', 'default': 'Living Expenses', 'width': 5}, {'key': 'amount', 'label': "Amount", 'type': 'number', 'default': 50000, 'width': 3}, {'key': 'growthRate', 'label': 'Growth', 'type': 'number', 'default': 3, 'width': 3}], 'Recurring Expenses', {'name': 'Living Expenses', 'amount': 50000, 'growthRate': 3})
    elif edit_section == "One-Time Events":
        create_dynamic_list_ui('oneTimeEvents', [{'key': 'name', 'label': 'Event Name', 'type': 'text', 'default': 'New Event', 'width': 4}, {'key': 'type', 'label': 'Type', 'type': 'select', 'options': ['Income', 'Expense'], 'default': 'Expense', 'width': 2}, {'key': 'amount', 'label': 'Amount ($)', 'type': 'number', 'default': 20000, 'width': 2}, {'key': 'year', 'label': 'Year', 'type': 'number', 'default': TODAY_YEAR + 15, 'width': 2}], 'One-Time Events', {'name': 'New Event', 'type': 'Expense', 'amount': 20000, 'year': TODAY_YEAR + 15})
    elif edit_section == "Market Volatility":
        create_dynamic_list_ui('marketCrashes', [{'key': 'startYear', 'label': 'Crash Start', 'type': 'number', 'default': TODAY_YEAR + 10, 'width': 2}, {'key': 'duration', 'label': 'Duration', 'type': 'number', 'default': 2, 'width': 2}, {'key': 'totalDecline', 'label': 'Decline (%)', 'type': 'number', 'default': 30, 'width': 2}, {'key': 'timing', 'label': 'Timing', 'type': 'select', 'options': ['start', 'end'], 'default': 'start', 'width': 2}], 'Market Volatility', {'startYear': TODAY_YEAR + 10, 'duration': 2, 'totalDecline': 30, 'timing': 'start'})

# --- Simulation Runner ---
if st.button("🚀 Run & Compare All Scenarios", type="primary", use_container_width=True):
    exchange_rate = get_exchange_rate()
    with st.spinner("Calculating all scenarios..."):
        st.session_state.results = [run_simulation(s, exchange_rate) for s in st.session_state.scenarios]

# --- Results Display ---
st.header("📊 Simulation Results")
if st.session_state.results:
    has_errors = any(res.get('errors') for res in st.session_state.results)
    if has_errors:
        for i, result in enumerate(st.session_state.results):
            if result.get('errors'):
                st.error(f"**Scenario '{st.session_state.scenarios[i]['name']}' has input errors:**")
                for error in result['errors']:
                    st.warning(f"- {error}")
    else:
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        symbols = ['circle', 'square', 'diamond', 'cross', 'x']
        summary_data = []

        for i, result in enumerate(st.session_state.results):
            scenario = st.session_state.scenarios[i]
            if result and result.get('data'):
                years = [d['year'] for d in result['data']]
                balances = [d['balance'] for d in result['data']]
                ages = [d['age'] for d in result['data']]
                
                fig.add_trace(go.Scatter(
                    x=years, y=balances, mode='lines+markers', name=scenario['name'],
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=7, symbol=symbols[i % len(symbols)]),
                    hovertext=[f"Age: {age}" for age in ages],
                    hovertemplate='<b>%{data.name}</b><br><b>Year:</b> %{x}<br><b>Balance:</b> %{y:$,.0f}<br><b>%{hovertext}</b><extra></extra>'
                ))
                
                final_balance = balances[-1] if balances else 0
                depletion_text = f"{result['depletion_year']} (Age: {result['depletion_year'] - scenario['birthYear']})" if result['depletion_year'] else "Sustained"
                summary_data.append({"Scenario": scenario['name'], "Final Balance": format_currency(final_balance), "Funds Depleted In": depletion_text})

        fig.update_layout(title="Retirement Portfolio Projection", xaxis_title="Year", yaxis_title="Portfolio Balance", yaxis_tickprefix="$", yaxis_tickformat="~s", legend_title="Scenarios", template="plotly_dark", height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h5>Results Summary</h5>", unsafe_allow_html=True)
        st.table(pd.DataFrame(summary_data).set_index("Scenario"))

        with st.expander("View Detailed Yearly Data"):
            selected_scenario_for_table = st.selectbox("Select scenario to view details", [s['name'] for s in st.session_state.scenarios])
            idx = [s['name'] for s in st.session_state.scenarios].index(selected_scenario_for_table)
            if st.session_state.results[idx] and st.session_state.results[idx].get('data'):
                df = pd.DataFrame(st.session_state.results[idx]['data'])
                df['balance'] = df['balance'].apply(format_currency)
                st.dataframe(df.set_index('year'), use_container_width=True)

else:
    st.info("Adjust settings in the expander above and click 'Run & Compare All Scenarios'.")
