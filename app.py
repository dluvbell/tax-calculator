import streamlit as st
import pandas as pd
import requests # For fetching exchange rates

# --- Tax Data for 2025 (Based on official sources and financial publications) ---
FED_BRACKETS_2025 = {
    57375: 0.15, 114750: 0.205, 177882: 0.26, 253414: 0.29, float('inf'): 0.33
}
ON_BRACKETS_2025 = {
    52886: 0.0505, 105775: 0.0915, 150000: 0.1116, 220000: 0.1216, float('inf'): 0.1316
}
FED_BPA_2025 = 16129
ON_BPA_2025 = 12399
OAS_CLAWBACK_THRESHOLD_2025 = 93454
OAS_CLAWBACK_RATE = 0.15
ELIGIBLE_DIVIDEND_GROSS_UP = 1.38
FED_ELIGIBLE_DIVIDEND_CREDIT_RATE = 0.150198
ON_ELIGIBLE_DIVIDEND_CREDIT_RATE = 0.10

# --- Exchange Rate Function ---
@st.cache_data(ttl=3600) # Cache the exchange rate for 1 hour
def get_exchange_rate():
    """Fetches the latest USD to CAD exchange rate from a free API."""
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        response.raise_for_status()
        data = response.json()
        return data['rates']['CAD']
    except requests.exceptions.RequestException as e:
        # Use a fallback rate if the API call fails
        return 1.35

def calculate_progressive_tax(income, brackets):
    """Helper function to calculate tax based on progressive brackets."""
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

def calculate_taxes_for_person(income_data):
    """Core function to calculate all taxes for a single person's income data."""
    gross_income = sum(v for k, v in income_data.items() if k not in ['us_dividend_account', 'us_dividends_usd'])
    
    eligible_dividends_actual = income_data.get('cdn_dividends', 0)
    grossed_up_dividends = eligible_dividends_actual * ELIGIBLE_DIVIDEND_GROSS_UP

    us_dividends_cad = income_data.get('us_dividends', 0) # This is now in CAD
    us_dividend_account = income_data.get('us_dividend_account', 'Non-Registered')
    
    taxable_us_dividends = 0
    us_withholding_tax = 0
    if us_dividend_account in ['Non-Registered', 'TFSA']:
        us_withholding_tax = us_dividends_cad * 0.15
        if us_dividend_account == 'Non-Registered':
            taxable_us_dividends = us_dividends_cad
    
    capital_gains_taxable = income_data.get('capital_gains', 0) * 0.5

    total_taxable_income = (
        income_data.get('cpp_oas', 0) + income_data.get('pension_rrif', 0) +
        income_data.get('other_income', 0) + income_data.get('interest', 0) +
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
        oas_clawback = min(oas_clawback, income_data.get('cpp_oas', 0))

    total_tax = federal_tax + provincial_tax + oas_clawback
    net_income = gross_income - total_tax

    return {"gross_income": gross_income, "total_taxable_income": total_taxable_income, "total_tax": total_tax, "net_income": net_income}

def render_income_inputs(person_key, exchange_rate):
    """Renders the set of income input fields for a person."""
    st.markdown(f"#### **{person_key}'s** Annual Income")

    cols = st.columns(2)
    with cols[0]:
        income_data = {
            'cpp_oas': st.number_input("Government Pensions (CPP/OAS)", min_value=0, value=15000, key=f"{person_key}_cpp"),
            'pension_rrif': st.number_input("Private/Work Pensions (RRIF)", min_value=0, value=20000, key=f"{person_key}_pension"),
            'interest': st.number_input("Interest Income", min_value=0, value=1000, key=f"{person_key}_interest"),
            'other_income': st.number_input("Other Income (Work/Business)", min_value=0, value=0, key=f"{person_key}_other")
        }
    with cols[1]:
        income_data['cdn_dividends'] = st.number_input("Eligible Canadian Dividends", min_value=0, value=5000, key=f"{person_key}_cdn_div")
        us_dividends_usd = st.number_input("US Dividends (USD)", min_value=0.0, value=2000.0, format="%.2f", key=f"{person_key}_us_div_usd", help="Enter the amount in US Dollars.")
        income_data['us_dividend_account'] = st.selectbox("US Dividend Account", ("Non-Registered", "RRSP/RRIF", "TFSA"), key=f"{person_key}_us_acct")
        income_data['capital_gains'] = st.number_input("Capital Gains", min_value=0, value=3000, key=f"{person_key}_cg")
    
    income_data['us_dividends'] = us_dividends_usd * exchange_rate
    income_data['us_dividends_usd'] = us_dividends_usd
    
    return income_data

st.set_page_config(layout="wide")
st.title("🇨🇦 After-Tax Income Calculator for Canadian Residents")
st.markdown("Calculates post-retirement after-tax income for couples based on various income sources, using **2025 tax rules**.")

exchange_rate = get_exchange_rate()
st.info(f"**Applied Exchange Rate:** 1 USD = {exchange_rate:.4f} CAD (Live Data)")

st.header("1. Enter Income Information")
you_inputs = render_income_inputs("You", exchange_rate)
spouse_inputs = render_income_inputs("Spouse", exchange_rate)

st.header("2. Tax Optimization Strategy")
optimize_pension = st.checkbox("**Apply Pension Income Splitting**", value=True, help="Automatically transfers up to 50% of eligible pension income from the higher-income spouse to the lower-income spouse to minimize the couple's total tax payable.")

st.header("3. Calculate Results")
if st.button("Calculate Net Income", type="primary", use_container_width=True):
    you_tax_no_split = calculate_taxes_for_person(you_inputs)
    spouse_tax_no_split = calculate_taxes_for_person(spouse_inputs)
    total_tax_no_split = you_tax_no_split['total_tax'] + spouse_tax_no_split['total_tax']

    you_tax_final, spouse_tax_final, tax_savings = you_tax_no_split, spouse_tax_no_split, 0

    if optimize_pension:
        min_total_tax = total_tax_no_split
        if you_tax_no_split['total_taxable_income'] > spouse_tax_no_split['total_taxable_income']:
            pension_to_split, from_person = you_inputs.get('pension_rrif', 0), "you"
        else:
            pension_to_split, from_person = spouse_inputs.get('pension_rrif', 0), "spouse"
        
        max_split = pension_to_split * 0.5
        for i in range(51):
            split_amount = max_split * (i / 50.0)
            temp_you, temp_spouse = you_inputs.copy(), spouse_inputs.copy()

            if from_person == "you":
                temp_you['pension_rrif'] -= split_amount
                temp_spouse['pension_rrif'] += split_amount
            else:
                temp_spouse['pension_rrif'] -= split_amount
                temp_you['pension_rrif'] += split_amount

            you_tax_temp = calculate_taxes_for_person(temp_you)
            spouse_tax_temp = calculate_taxes_for_person(temp_spouse)
            current_total_tax = you_tax_temp['total_tax'] + spouse_tax_temp['total_tax']

            if current_total_tax < min_total_tax:
                min_total_tax = current_total_tax
                you_tax_final, spouse_tax_final = you_tax_temp, spouse_tax_temp
        
        tax_savings = total_tax_no_split - min_total_tax

    st.subheader("Results Summary")
    total_gross = you_tax_final['gross_income'] + spouse_tax_final['gross_income']
    final_total_tax = you_tax_final['total_tax'] + spouse_tax_final['total_tax']
    final_net = you_tax_final['net_income'] + spouse_tax_final['net_income']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Combined Gross Income", f"${total_gross:,.0f}")
    c2.metric("Combined Total Tax", f"${final_total_tax:,.0f}")
    c3.metric("Combined After-Tax Income", f"${final_net:,.0f}")
    c4.metric("Average Monthly Net Income", f"${(final_net / 12):,.0f}")

    if optimize_pension and tax_savings > 0.01:
        st.success(f"**Tax Savings:** You saved **${tax_savings:,.0f}** by applying pension income splitting!")

    with st.expander("View Detailed Results"):
        df = pd.DataFrame({
            "Item": ["Gross Income", "Total Tax", "**After-Tax Income (Net)**"],
            "You": [f"${you_tax_final['gross_income']:,.0f}", f"${you_tax_final['total_tax']:,.0f}", f"**${you_tax_final['net_income']:,.0f}**"],
            "Spouse": [f"${spouse_tax_final['gross_income']:,.0f}", f"${spouse_tax_final['total_tax']:,.0f}", f"**${spouse_tax_final['net_income']:,.0f}**"],
            "Total": [f"${total_gross:,.0f}", f"${final_total_tax:,.0f}", f"**${final_net:,.0f}**"]
        }).set_index("Item")
        st.dataframe(df, use_container_width=True)

    st.header("4. Link with Main Retirement Calculator")
    if st.button("Send this result to the Retirement Calculator", use_container_width=True):
        st.session_state['net_income_from_calc'] = final_net
        st.success(f"✅ **${final_net:,.0f}** has been sent to the Retirement Calculator.")
