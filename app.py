import streamlit as st
import pandas as pd
import requests # 환율 정보를 위해 추가

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

# --- 환율 가져오기 기능 ---
@st.cache_data(ttl=3600) # 1시간 동안 환율 정보 캐싱
def get_exchange_rate():
    """Fetches the latest USD to CAD exchange rate from a free API."""
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        response.raise_for_status()
        data = response.json()
        return data['rates']['CAD']
    except requests.exceptions.RequestException as e:
        # API 호출 실패 시 기본값 사용
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
    if us_dividend_account in ['Non-Registered', 'TFSA']:
        us_withholding_tax = us_dividends_cad * 0.15
        if us_dividend_account == 'Non-Registered':
            taxable_us_dividends = us_dividends_cad
    else: # RRSP/RRIF
        us_withholding_tax = 0

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
    st.markdown(f"#### **{person_key}**의 연간 소득 (Annual Income)")

    cols = st.columns(2)
    with cols[0]:
        income_data = {
            'cpp_oas': st.number_input("정부 연금 (CPP/OAS)", min_value=0, value=15000, key=f"{person_key}_cpp"),
            'pension_rrif': st.number_input("개인/직장 연금 (Pension/RRIF)", min_value=0, value=20000, key=f"{person_key}_pension"),
            'interest': st.number_input("이자 소득 (Interest)", min_value=0, value=1000, key=f"{person_key}_interest"),
            'other_income': st.number_input("기타 소득 (근로/사업 등)", min_value=0, value=0, key=f"{person_key}_other")
        }
    with cols[1]:
        income_data['cdn_dividends'] = st.number_input("캐나다 배당금 (Eligible)", min_value=0, value=5000, key=f"{person_key}_cdn_div")
        us_dividends_usd = st.number_input("미국 배당금 (USD)", min_value=0.0, value=2000.0, format="%.2f", key=f"{person_key}_us_div_usd", help="미국 달러(USD)로 금액을 입력하세요.")
        income_data['us_dividend_account'] = st.selectbox("미국 배당금 수령 계좌", ("Non-Registered", "RRSP/RRIF", "TFSA"), key=f"{person_key}_us_acct")
        income_data['capital_gains'] = st.number_input("자본 이득 (Capital Gains)", min_value=0, value=3000, key=f"{person_key}_cg")
    
    income_data['us_dividends'] = us_dividends_usd * exchange_rate
    income_data['us_dividends_usd'] = us_dividends_usd
    
    return income_data

st.set_page_config(layout="wide")
st.title("🇨🇦 캐나다 거주자용 세후 실수령액 계산기")
st.markdown("은퇴 후 다양한 소득원에 대한 **2025년 기준** 세금을 계산하여, 부부의 최종 실수령액을 예측합니다.")

exchange_rate = get_exchange_rate()
st.info(f"**적용 환율:** 1 USD = {exchange_rate:.4f} CAD (실시간 정보)")

st.header("1. 소득 정보 입력")
you_inputs = render_income_inputs("본인", exchange_rate)
spouse_inputs = render_income_inputs("배우자", exchange_rate)

st.header("2. 세금 최적화 전략")
optimize_pension = st.checkbox("**연금 소득 분할 (Pension Income Splitting) 적용하기**", value=True)

st.header("3. 계산 결과 확인")
if st.button("실수령액 계산하기", type="primary", use_container_width=True):
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

    st.subheader("결과 요약")
    total_gross = you_tax_final['gross_income'] + spouse_tax_final['gross_income']
    final_total_tax = you_tax_final['total_tax'] + spouse_tax_final['total_tax']
    final_net = you_tax_final['net_income'] + spouse_tax_final['net_income']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("부부 합산 총소득", f"${total_gross:,.0f}")
    c2.metric("부부 합산 총납부 세금", f"${final_total_tax:,.0f}")
    c3.metric("부부 합산 세후 실수령액", f"${final_net:,.0f}")
    c4.metric("월 평균 실수령액", f"${(final_net / 12):,.0f}")

    if optimize_pension and tax_savings > 0.01:
        st.success(f"**세금 최적화 효과:** 연금 소득 분할을 통해 **${tax_savings:,.0f}**의 세금을 절약했습니다!")

    with st.expander("상세 결과 보기"):
        df = pd.DataFrame({
            "항목": ["총소득 (Gross)", "총납부 세금 (Tax)", "**세후 실수령액 (Net)**"],
            "본인 (You)": [f"${you_tax_final['gross_income']:,.0f}", f"${you_tax_final['total_tax']:,.0f}", f"**${you_tax_final['net_income']:,.0f}**"],
            "배우자 (Spouse)": [f"${spouse_tax_final['gross_income']:,.0f}", f"${spouse_tax_final['total_tax']:,.0f}", f"**${spouse_tax_final['net_income']:,.0f}**"],
            "부부 합산 (Total)": [f"${total_gross:,.0f}", f"${final_total_tax:,.0f}", f"**${final_net:,.0f}**"]
        }).set_index("항목")
        st.dataframe(df, use_container_width=True)

    st.header("4. 메인 은퇴 계산기와 연동")
    if st.button("이 결과를 메인 은퇴 계산기에 보내기", use_container_width=True):
        st.session_state['net_income_from_calc'] = final_net
        st.success(f"✅ **${final_net:,.0f}**이(가) 은퇴 계산기로 전송되었습니다.")
