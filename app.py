# Note: File handling and visualization operations are only supported in Python in this execution environment. For best results with file analysis and related tasks, we recommend using our tool: https://analyst.galaxy.ai/ ,You can request code in other languages in the chat, but execution requires Python for these features.

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

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        # Sandbox environment has no internet access—this section will not work here.
        # Using default value instead
        return 1.35
    except Exception:
        return 1.35

def format_currency(amount):
    return f"${amount:,.0f}"

# --- Core Tax Calculation Logic ---
def calculate_progressive_tax(income, brackets):
    """Calculate tax based on progressive brackets"""
    if income <= 0:
        return 0
    
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
    """Calculate after-tax income with improved error handling"""
    gross_income = sum(yearly_income_details.values())
    
    # Handle dividend calculations
    eligible_dividends_actual = yearly_income_details.get('cdn_dividends', 0)
    grossed_up_dividends = eligible_dividends_actual * ELIGIBLE_DIVIDEND_GROSS_UP
    us_dividends_cad = yearly_income_details.get('us_dividends', 0)
    
    # US dividend withholding tax
    taxable_us_dividends = 0
    us_withholding_tax = 0
    if us_dividend_account in ['Non-Registered', 'TFSA']:
        us_withholding_tax = us_dividends_cad * 0.15
        if us_dividend_account == 'Non-Registered':
            taxable_us_dividends = us_dividends_cad
    
    # Capital gains inclusion
    capital_gains_taxable = yearly_income_details.get('capital_gains', 0) * 0.5

    # Calculate total taxable income
    total_taxable_income = (
        yearly_income_details.get('cpp_oas', 0) + 
        yearly_income_details.get('pension_rrif', 0) +
        yearly_income_details.get('other_income', 0) + 
        yearly_income_details.get('interest', 0) +
        grossed_up_dividends + 
        taxable_us_dividends + 
        capital_gains_taxable
    )

    # Calculate taxes before credits
    federal_tax_before_credits = calculate_progressive_tax(total_taxable_income, FED_BRACKETS_2025)
    provincial_tax_before_credits = calculate_progressive_tax(total_taxable_income, ON_BRACKETS_2025)
    
    # Apply basic personal amounts
    federal_tax = federal_tax_before_credits
    provincial_tax = provincial_tax_before_credits
    
    federal_tax -= min(total_taxable_income, FED_BPA_2025) * list(FED_BRACKETS_2025.values())[0]
    provincial_tax -= min(total_taxable_income, ON_BPA_2025) * list(ON_BRACKETS_2025.values())[0]

    # Apply dividend tax credits
    federal_tax -= grossed_up_dividends * FED_ELIGIBLE_DIVIDEND_CREDIT_RATE
    provincial_tax -= grossed_up_dividends * ON_ELIGIBLE_DIVIDEND_CREDIT_RATE

    # Foreign tax credit for US dividends
    if us_dividend_account == 'Non-Registered' and total_taxable_income > 0:
        canadian_tax_on_us_income = taxable_us_dividends * (federal_tax_before_credits / total_taxable_income)
        foreign_tax_credit = min(us_withholding_tax, canadian_tax_on_us_income)
        federal_tax -= foreign_tax_credit

    # Ensure taxes are not negative
    federal_tax = max(0, federal_tax)
    provincial_tax = max(0, provincial_tax)

    # OAS clawback calculation
    net_income_for_clawback = total_taxable_income - (grossed_up_dividends - eligible_dividends_actual)
    oas_clawback = 0
    if net_income_for_clawback > OAS_CLAWBACK_THRESHOLD_2025:
        oas_clawback = (net_income_for_clawback - OAS_CLAWBACK_THRESHOLD_2025) * OAS_CLAWBACK_RATE
        oas_clawback = min(oas_clawback, yearly_income_details.get('cpp_oas', 0))

    total_tax = federal_tax + provincial_tax + oas_clawback + us_withholding_tax
    net_income = gross_income - total_tax
    
    return net_income

# --- Core Retirement Simulation Engine ---
def run_simulation(scenario, exchange_rate):
    """Run retirement simulation with improved error handling"""
    investment = float(scenario['initialInvestment'])
    years_to_grow = scenario['startYear'] - TODAY_YEAR
    
    # Pre-retirement growth
    if years_to_grow > 0:
        investment *= (1 + float(scenario['investmentReturn']) / 100) ** years_to_grow

    yearly_data = []
    depletion_year = None

    for year in range(scenario['startYear'], scenario['endYear'] + 1):
        # Check for depletion
        if investment <= 0 and depletion_year is None:
            depletion_year = year
        
        balance = max(0, investment)
        age = year - scenario['birthYear']
        
        # Apply market crashes at start of year
        for crash in scenario.get('marketCrashes', []):
            duration = int(crash.get('duration', 1))
            if duration > 0 and crash.get('timing') == 'start':
                crash_start = int(crash['startYear'])
                if year >= crash_start and year < crash_start + duration:
                    decline_rate = (1 - float(crash['totalDecline']) / 100) ** (1 / duration)
                    investment *= decline_rate

        # Apply one-time events
        for event in scenario.get('oneTimeEvents', []):
            if int(event['year']) == year:
                amount = float(event['amount'])
                investment += amount if event['type'] == 'Income' else -amount

        # Calculate yearly income
        yearly_income_details = {}
        for income in scenario['incomes']:
            if year >= int(income['startYear']):
                years_from_today = int(income['startYear']) - TODAY_YEAR
                amount_at_start = float(income['amount']) * ((1 + float(income['growthRate']) / 100) ** max(0, years_from_today))
                years_since_start = year - int(income['startYear'])
                future_value = amount_at_start * ((1 + float(income['growthRate']) / 100) ** years_since_start)
                income_type = income.get('type', 'other_income')
                yearly_income_details[income_type] = yearly_income_details.get(income_type, 0) + future_value
        
        # Calculate net income and expenses
        net_annual_income = calculate_after_tax_income(yearly_income_details, scenario['us_dividend_account'])
        net_annual_expense = sum(
            float(exp['amount']) * ((1 + float(exp['growthRate']) / 100) ** (year - TODAY_YEAR)) 
            for exp in scenario['expenses']
        )
        
        # Update investment balance
        investment += (net_annual_income - net_annual_expense)
        
        # Apply investment returns
        if investment > 0:
            investment *= (1 + float(scenario['investmentReturn']) / 100)

        # Apply market crashes at end of year
        for crash in scenario.get('marketCrashes', []):
            duration = int(crash.get('duration', 1))
            if duration > 0 and crash.get('timing') == 'end':
                crash_start = int(crash['startYear'])
                if year >= crash_start and year < crash_start + duration:
                    decline_rate = (1 - float(crash['totalDecline']) / 100) ** (1 / duration)
                    investment *= decline_rate
        
        # Store year data
        yearly_data.append({
            'year': year, 
            'age': age, 
            'balance': round(balance),
            'net_income': round(net_annual_income),
            'expenses': round(net_annual_expense),
            'net_cash_flow': round(net_annual_income - net_annual_expense)
        })
            
    return {'data': yearly_data, 'depletion_year': depletion_year}

# --- UI Components & Callbacks ---
def add_item(list_name, default_item):
    st.session_state.scenarios[st.session_state.active_scenario_index][list_name].append(copy.deepcopy(default_item))

def delete_item(list_name, index):
    st.session_state.scenarios[st.session_state.active_scenario_index][list_name].pop(index)

def create_dynamic_list_ui(list_name, fields, title, default_item):
    """Create dynamic list UI with improved mobile responsiveness"""
    st.markdown(f"<h5>{title}</h5>", unsafe_allow_html=True)
    active_scenario = st.session_state.scenarios[st.session_state.active_scenario_index]
    
    if list_name not in active_scenario: 
        active_scenario[list_name] = []

    # Mobile-friendly layout
    for i, item in enumerate(active_scenario[list_name]):
        with st.container():
            # Use expander for mobile-friendly view
            with st.expander(f"{item.get('name', f'Item {i+1}')}", expanded=True):
                # Create responsive columns
                if st.session_state.get('mobile_view', False):
                    # Stack fields vertically on mobile
                    for field in fields:
                        if field['type'] == 'text':
                            item[field['key']] = st.text_input(
                                field['label'], 
                                value=item[field['key']], 
                                key=f"{list_name}_{i}_{field['key']}"
                            )
                        elif field['type'] == 'number':
                            item[field['key']] = st.number_input(
                                field['label'], 
                                value=item[field['key']], 
                                key=f"{list_name}_{i}_{field['key']}"
                            )
                        elif field['type'] == 'select':
                            item[field['key']] = st.selectbox(
                                field['label'], 
                                field['options'], 
                                index=field['options'].index(item[field['key']]), 
                                key=f"{list_name}_{i}_{field['key']}"
                            )
                else:
                    # Use columns for desktop
                    cols = st.columns([f['width'] for f in fields] + [1])
                    for j, field in enumerate(fields):
                        if field['type'] == 'text':
                            item[field['key']] = cols[j].text_input(
                                field['label'], 
                                value=item[field['key']], 
                                key=f"{list_name}_{i}_{field['key']}"
                            )
                        elif field['type'] == 'number':
                            item[field['key']] = cols[j].number_input(
                                field['label'], 
                                value=item[field['key']], 
                                key=f"{list_name}_{i}_{field['key']}"
                            )
                        elif field['type'] == 'select':
                            item[field['key']] = cols[j].selectbox(
                                field['label'], 
                                field['options'], 
                                index=field['options'].index(item[field['key']]), 
                                key=f"{list_name}_{i}_{field['key']}"
                            )
                    
                    cols[-1].button(
                        "🗑️", 
                        key=f"{list_name}_del_{i}", 
                        help=f"Remove this item", 
                        on_click=delete_item, 
                        args=(list_name, i)
                    )

    st.button(
        f"➕ Add {title.replace('Recurring ','').replace('s','')}", 
        key=f"add_{list_name}", 
        use_container_width=True, 
        on_click=add_item, 
        args=(list_name, copy.deepcopy(default_item))
    )

# --- Main App Layout ---
st.set_page_config(
    page_title="Retirement & Tax Planner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for better mobile responsiveness
st.markdown("""
<style>
    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .stButton > button {
            width: 100%;
            margin: 2px 0;
        }
        .row-widget.stSelectbox {
            width: 100%;
        }
        .row-widget.stNumberInput {
            width: 100%;
        }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
        }
    }
    
    /* Improve number input appearance */
    .stNumberInput > div > div > input {
        text-align: right;
    }
    
    /* Better button styling */
    .stButton > button {
        border-radius: 5px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Detect mobile view (simplified approach)
if 'mobile_view' not in st.session_state:
    st.session_state.mobile_view = False

st.title("📈 Integrated Retirement & Tax Planner")
st.markdown("Advanced retirement planning with detailed Canadian tax calculations for 2025")

# --- Initialize Session State ---
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = [{
        'name': 'My Retirement Plan', 
        'initialInvestment': 500000, 
        'investmentReturn': 6, 
        'birthYear': 1960, 
        'startYear': TODAY_YEAR + 5, 
        'endYear': TODAY_YEAR + 35, 
        'us_dividend_account': 'Non-Registered', 
        'incomes': [
            {'name': 'CPP/OAS', 'amount': 15000, 'startYear': TODAY_YEAR + 5, 'growthRate': 2.5, 'type': 'cpp_oas'},
            {'name': 'Pension', 'amount': 20000, 'startYear': TODAY_YEAR + 5, 'growthRate': 2.0, 'type': 'pension_rrif'}
        ], 
        'expenses': [
            {'name': 'Living Expenses', 'amount': 60000, 'growthRate': 3.0}
        ], 
        'oneTimeEvents': [], 
        'marketCrashes': []
    }]

if 'active_scenario_index' not in st.session_state:
    st.session_state.active_scenario_index = 0
if 'results' not in st.session_state:
    st.session_state.results = None

# --- Settings Panel ---
with st.expander("⚙️ Settings & Inputs", expanded=True):
    # Scenario Manager
    st.markdown("<h4>📋 Scenario Manager</h4>", unsafe_allow_html=True)
    
    def update_active_index():
        st.session_state.active_scenario_index = st.session_state.scenario_selector

    def add_scenario_cb():
        new_scenario = {
            'name': f'Scenario {len(st.session_state.scenarios) + 1}', 
            'initialInvestment': 500000, 
            'investmentReturn': 6, 
            'birthYear': 1960, 
            'startYear': TODAY_YEAR + 5, 
            'endYear': TODAY_YEAR + 35, 
            'us_dividend_account': 'Non-Registered', 
            'incomes': [], 
            'expenses': [], 
            'oneTimeEvents': [], 
            'marketCrashes': []
        }
        st.session_state.scenarios.append(new_scenario)
        st.session_state.active_scenario_index = len(st.session_state.scenarios) - 1

    def copy_scenario_cb():
        scenario_to_copy = copy.deepcopy(st.session_state.scenarios[st.session_state.active_scenario_index])
        scenario_to_copy['name'] = f"{scenario_to_copy['name']} (Copy)"
        st.session_state.scenarios.append(scenario_to_copy)
        st.session_state.active_scenario_index = len(st.session_state.scenarios) - 1

    def delete_scenario_cb():
        if len(st.session_state.scenarios) > 1:
            st.session_state.scenarios.pop(st.session_state.active_scenario_index)
            st.session_state.active_scenario_index = 0

    scenario_names = [s['name'] for s in st.session_state.scenarios]
    
    # Scenario selector with buttons
    sc_cols = st.columns([3, 1, 1, 1])
    with sc_cols[0]:
        st.selectbox(
            "Active Scenario", 
            options=range(len(scenario_names)), 
            format_func=lambda x: scenario_names[x], 
            index=st.session_state.active_scenario_index, 
            key="scenario_selector", 
            on_change=update_active_index,
            label_visibility="collapsed"
        )
    
    sc_cols[1].button("➕ New", use_container_width=True, disabled=len(st.session_state.scenarios) >= 5, on_click=add_scenario_cb)
    sc_cols[2].button("📋 Copy", use_container_width=True, disabled=len(st.session_state.scenarios) >= 5, on_click=copy_scenario_cb)
    sc_cols[3].button("🗑️ Delete", use_container_width=True, disabled=len(st.session_state.scenarios) <= 1, on_click=delete_scenario_cb)
    
    st.markdown("---")
    
    active_scenario = st.session_state.scenarios[st.session_state.active_scenario_index]

    # Section selector
    edit_section = st.selectbox(
        "📝 Edit Section", 
        ["General & Tax Settings", "Recurring Incomes & Expenses", "Events & Market Volatility"],
        help="Select which section to edit"
    )
    
    if edit_section == "General & Tax Settings":
        active_scenario['name'] = st.text_input("Scenario Name", value=active_scenario['name'])
        
        # Basic settings
        st.markdown("##### Basic Settings")
        cols = st.columns(2)
        with cols[0]:
            active_scenario['initialInvestment'] = st.number_input(
                "Initial Investment ($)", 
                value=active_scenario['initialInvestment'], 
                format="%d",
                step=10000,
                help="Starting portfolio value"
            )
            active_scenario['birthYear'] = st.number_input(
                "Birth Year", 
                value=active_scenario['birthYear'], 
                format="%d",
                min_value=1940,
                max_value=2000
            )
            active_scenario['startYear'] = st.number_input(
                "Retirement Start Year", 
                value=active_scenario['startYear'], 
                format="%d",
                min_value=TODAY_YEAR,
                max_value=TODAY_YEAR + 50
            )
        
        with cols[1]:
            active_scenario['investmentReturn'] = st.number_input(
                "Average Annual Return (%)", 
                value=active_scenario['investmentReturn'],
                min_value=0.0,
                max_value=20.0,
                step=0.5,
                help="Expected annual investment return"
            )
            active_scenario['endYear'] = st.number_input(
                "Planning End Year", 
                value=active_scenario['endYear'], 
                format="%d",
                min_value=active_scenario['startYear'] + 1,
                max_value=TODAY_YEAR + 100
            )
        
        # Tax settings
        st.markdown("##### Tax Settings")
        active_scenario['us_dividend_account'] = st.selectbox(
            "US Dividend Account Type",
            ("Non-Registered", "RRSP/RRIF", "TFSA"),
            index=["Non-Registered", "RRSP/RRIF", "TFSA"].index(active_scenario['us_dividend_account']),
            help="Account type affects US dividend withholding tax treatment"
        )
        
        # Info box
        st.info("💡 Tax calculations use 2025 Canadian federal and Ontario provincial rates")

    elif edit_section == "Recurring Incomes & Expenses":
        # Income types with proper categorization
        income_types = [
            ('cpp_oas', 'CPP/OAS'),
            ('pension_rrif', 'Pension/RRIF'),
            ('cdn_dividends', 'Canadian Dividends'),
            ('us_dividends', 'US Dividends'),
            ('interest', 'Interest Income'),
            ('capital_gains', 'Capital Gains'),
            ('other_income', 'Other Income')
        ]
        
        # Create tabs for better organization
        income_tab, expense_tab = st.tabs(["💰 Incomes", "💸 Expenses"])
        
        with income_tab:
            st.markdown("##### Recurring Income Sources")
            
            # Add income type selector for new incomes
            if 'new_income_type' not in st.session_state:
                st.session_state.new_income_type = 'other_income'
            
            # Income list
            for i, income in enumerate(active_scenario.get('incomes', [])):
                with st.expander(f"📥 {income.get('name', 'Income')} - ${income.get('amount', 0):,.0f}/year", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        income['name'] = st.text_input("Name", value=income['name'], key=f"income_name_{i}")
                        income['amount'] = st.number_input("Annual Amount ($)", value=income['amount'], step=1000, key=f"income_amount_{i}")
                        income['type'] = st.selectbox(
                            "Income Type",
                            options=[t[0] for t in income_types],
                            format_func=lambda x: dict(income_types)[x],
                            index=[t[0] for t in income_types].index(income.get('type', 'other_income')),
                            key=f"income_type_{i}",
                            help="Tax treatment varies by income type"
                        )
                    with col2:
                        income['startYear'] = st.number_input("Start Year", value=income['startYear'], min_value=TODAY_YEAR, key=f"income_start_{i}")
                        income['growthRate'] = st.number_input("Annual Growth Rate (%)", value=income['growthRate'], step=0.5, key=f"income_growth_{i}")
                        if st.button("🗑️ Remove", key=f"del_income_{i}"):
                            active_scenario['incomes'].pop(i)
                            st.rerun()
            
            # Add new income
            if st.button("➕ Add Income Source", use_container_width=True):
                new_income = {
                    'name': 'New Income',
                    'amount': 10000,
                    'startYear': TODAY_YEAR + 5,
                    'growthRate': 2.5,
                    'type': 'other_income'
                }
                active_scenario['incomes'].append(new_income)
                st.rerun()
        
        with expense_tab:
            st.markdown("##### Recurring Expenses")
            
            # Expense list
            for i, expense in enumerate(active_scenario.get('expenses', [])):
                with st.expander(f"📤 {expense.get('name', 'Expense')} - ${expense.get('amount', 0):,.0f}/year", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        expense['name'] = st.text_input("Name", value=expense['name'], key=f"expense_name_{i}")
                        expense['amount'] = st.number_input("Annual Amount ($)", value=expense['amount'], step=1000, key=f"expense_amount_{i}")
                    with col2:
                        expense['growthRate'] = st.number_input("Annual Growth Rate (%)", value=expense['growthRate'], step=0.5, key=f"expense_growth_{i}")
                        if st.button("🗑️ Remove", key=f"del_expense_{i}"):
                            active_scenario['expenses'].pop(i)
                            st.rerun()
            
            # Add new expense
            if st.button("➕ Add Expense", use_container_width=True):
                new_expense = {
                    'name': 'New Expense',
                    'amount': 50000,
                    'growthRate': 3.0
                }
                active_scenario['expenses'].append(new_expense)
                st.rerun()

    elif edit_section == "Events & Market Volatility":
        event_tab, crash_tab = st.tabs(["📅 One-Time Events", "📉 Market Crashes"])
        
        with event_tab:
            st.markdown("##### One-Time Events")
            
            # Event list
            for i, event in enumerate(active_scenario.get('oneTimeEvents', [])):
                with st.expander(f"{'💵' if event['type'] == 'Income' else '💸'} {event.get('name', 'Event')} - ${event.get('amount', 0):,.0f} in {event.get('year', TODAY_YEAR)}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        event['name'] = st.text_input("Event Name", value=event['name'], key=f"event_name_{i}")
                        event['type'] = st.selectbox("Type", ['Income', 'Expense'], index=['Income', 'Expense'].index(event['type']), key=f"event_type_{i}")
                    with col2:
                        event['amount'] = st.number_input("Amount ($)", value=event['amount'], step=1000, key=f"event_amount_{i}")
                        event['year'] = st.number_input("Year", value=event['year'], min_value=TODAY_YEAR, key=f"event_year_{i}")
                    
                    if st.button("🗑️ Remove", key=f"del_event_{i}"):
                        active_scenario['oneTimeEvents'].pop(i)
                        st.rerun()
            
            # Add new event
            if st.button("➕ Add Event", use_container_width=True):
                new_event = {
                    'name': 'New Event',
                    'type': 'Expense',
                    'amount': 20000,
                    'year': TODAY_YEAR + 10
                }
                active_scenario['oneTimeEvents'].append(new_event)
                st.rerun()
        
        with crash_tab:
            st.markdown("##### Market Volatility Events")
            st.info("💡 Model market downturns to stress-test your plan")
            
            # Crash list
            for i, crash in enumerate(active_scenario.get('marketCrashes', [])):
                with st.expander(f"📉 Market Crash {i+1}: {crash.get('totalDecline', 0)}% decline starting {crash.get('startYear', TODAY_YEAR)}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        crash['startYear'] = st.number_input("Start Year", value=crash['startYear'], min_value=TODAY_YEAR, key=f"crash_start_{i}")
                        crash['totalDecline'] = st.number_input("Total Decline (%)", value=crash['totalDecline'], min_value=0, max_value=100, key=f"crash_decline_{i}")
                    with col2:
                        crash['duration'] = st.number_input("Duration (years)", value=crash['duration'], min_value=1, max_value=5, key=f"crash_duration_{i}")
                        crash['timing'] = st.selectbox("Apply at", ['start', 'end'], index=['start', 'end'].index(crash['timing']), key=f"crash_timing_{i}", help="Apply decline at start or end of year")
                    
                    if st.button("🗑️ Remove", key=f"del_crash_{i}"):
                        active_scenario['marketCrashes'].pop(i)
                        st.rerun()
            
            # Add new crash
            if st.button("➕ Add Market Crash", use_container_width=True):
                new_crash = {
                    'startYear': TODAY_YEAR + 10,
                    'duration': 2,
                    'totalDecline': 30,
                    'timing': 'start'
                }
                active_scenario['marketCrashes'].append(new_crash)
                st.rerun()

# --- Run Simulation Button ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Run Simulation & Compare Scenarios", type="primary", use_container_width=True):
        exchange_rate = get_exchange_rate()
        with st.spinner("Running simulations..."):
            st.session_state.results = [run_simulation(s, exchange_rate) for s in st.session_state.scenarios]

# --- Results Display ---
if st.session_state.results:
    st.header("📊 Simulation Results")
    
    # Create visualization
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    symbols = ['circle', 'square', 'diamond', 'cross', 'x']
    summary_data = []

    for i, result in enumerate(st.session_state.results):
        scenario = st.session_state.scenarios[i]
        if result and result['data']:
            years = [d['year'] for d in result['data']]
            balances = [d['balance'] for d in result['data']]
            ages = [d['age'] for d in result['data']]
            
            # Add trace to plot
            fig.add_trace(go.Scatter(
                x=years, 
                y=balances, 
                mode='lines+markers', 
                name=scenario['name'],
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8, symbol=symbols[i % len(symbols)]),
                hovertext=[f"Age: {age}<br>Net Income: ${d['net_income']:,.0f}<br>Expenses: ${d['expenses']:,.0f}" 
                          for age, d in zip(ages, result['data'])],
                hovertemplate='<b>%{data.name}</b><br>' +
                             '<b>Year:</b> %{x}<br>' +
                             '<b>Balance:</b> %{y:$,.0f}<br>' +
                             '%{hovertext}<extra></extra>'
            ))
            
            # Calculate summary statistics
            final_balance = balances[-1] if balances else 0
            depletion_text = f"{result['depletion_year']} (Age {result['depletion_year'] - scenario['birthYear']})" if result['depletion_year'] else "Never"
            
            # Calculate success rate (simplified)
            success_rate = 100 if not result['depletion_year'] else max(0, ((result['depletion_year'] - scenario['startYear']) / (scenario['endYear'] - scenario['startYear'])) * 100)
            
            summary_data.append({
                "Scenario": scenario['name'], 
                "Final Balance": format_currency(final_balance), 
                "Depletion": depletion_text,
                "Success Rate": f"{success_rate:.0f}%"
            })

    # Update plot layout
    fig.update_layout(
        title={
            'text': "Retirement Portfolio Projection",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Year", 
        yaxis_title="Portfolio Balance ($)",
        yaxis_tickprefix="$", 
        yaxis_tickformat=",.0f",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        height=500,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Add grid
    fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.markdown("### 📋 Summary Comparison")
    summary_df = pd.DataFrame(summary_data)
    
    # Style the dataframe
    styled_df = summary_df.style.set_properties(**{
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]}
    ])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Detailed data viewer
    with st.expander("📊 View Detailed Yearly Data"):
        selected_scenario = st.selectbox(
            "Select scenario for detailed view", 
            [s['name'] for s in st.session_state.scenarios]
        )
        idx = [s['name'] for s in st.session_state.scenarios].index(selected_scenario)
        
        if st.session_state.results[idx]:
            df = pd.DataFrame(st.session_state.results[idx]['data'])
            
            # Format columns
            df['balance'] = df['balance'].apply(lambda x: f"${x:,.0f}")
            df['net_income'] = df['net_income'].apply(lambda x: f"${x:,.0f}")
            df['expenses'] = df['expenses'].apply(lambda x: f"${x:,.0f}")
            df['net_cash_flow'] = df['net_cash_flow'].apply(lambda x: f"${x:,.0f}")
            
            # Display with better column names
            df.columns = ['Year', 'Age', 'Portfolio Balance', 'After-Tax Income', 'Expenses', 'Net Cash Flow']
            st.dataframe(df.set_index('Year'), use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{selected_scenario}_retirement_projection.csv",
                mime="text/csv"
            )

    # Tax optimization tips
    with st.expander("💡 Tax Optimization Tips"):
        st.markdown("""
        Based on [schwab.com](https://www.schwab.com/learn/story/5-step-tax-smart-retirement-income-plan) and [turbotax.intuit.com](https://turbotax.intuit.com/tax-tips/retirement/tax-tips-after-you-retire/L6DBVFZ25):
        
        1. **Withdrawal Order Matters**: Consider withdrawing from taxable accounts first, then tax-deferred, and finally tax-free accounts
        2. **Manage Your Tax Brackets**: Keep income below key thresholds to minimize taxes and OAS clawback
        3. **Split Income**: Use pension splitting with your spouse to reduce overall taxes
        4. **Time Your RRSP/RRIF Conversions**: Convert strategically to manage tax brackets
        5. **Consider Tax-Efficient Investments**: Canadian eligible dividends and capital gains receive preferential tax treatment
        
        For more advanced strategies, consult with a qualified financial advisor or visit [ameriprise.com](https://www.ameriprise.com/financial-goals-priorities/taxes/how-to-minimize-taxes).
        """)

else:
    # Welcome message
    st.info("👆 Configure your retirement scenarios above and click 'Run Simulation' to see projections")
    
    # Quick tips
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        1. **Set Your Basic Info**: Enter your current savings, expected return, and retirement timeline
        2. **Add Income Sources**: Include CPP/OAS, pensions, and investment income
        3. **Add Expenses**: Enter your expected retirement expenses with inflation
        4. **Test Scenarios**: Create multiple scenarios to compare different strategies
        5. **Add Market Events**: Test your plan against market downturns
        6. **Run Simulation**: Click the button to see your retirement projection
        
        💡 **Pro Tip**: Create a conservative scenario with lower returns and higher expenses to stress-test your plan!
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    📊 Retirement & Tax Planner v2.0 | Tax rates current as of 2025 | 
    <a href="https://www.boldin.com/retirement/release-notes/" target="_blank">Learn more about retirement planning</a>
</div>
""", unsafe_allow_html=True)
