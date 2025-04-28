{\rtf1\ansi\ansicpg1252\cocoartf2580
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 AppleColorEmoji;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import numpy as np\
import pandas as pd\
import numpy_financial as npf\
import matplotlib.pyplot as plt\
\
st.set_page_config(page_title="ISO Residual Buyout Model", layout="wide")\
\
st.title("\\ud83d\\udcca ISO Residual Buyout Model - Advanced Version")\
\
# --- Sidebar Inputs ---\
st.sidebar.header("\\ud83d\\udee0\\ufe0f Model Inputs")\
\
# Basic Inputs\
portfolio_residual_monthly = st.sidebar.number_input("Starting Residual Inflow per Month ($)", value=50000)\
upfront_multiple = st.sidebar.number_input("Upfront Multiple (e.g., 4 for 4x)", value=34.0)\
annual_attrition_rate = st.sidebar.number_input("Starting Annual Attrition Rate (%)", value=15.0) / 100\
attrition_step_up_rate = st.sidebar.number_input("Attrition Step-Up per Year (%)", value=0.0) / 100\
guarantee_period_months = st.sidebar.number_input("Guarantee Period (Months)", value=0)\
\
# Earn-Outs\
st.sidebar.subheader("Earn-Out Structure")\
earn_outs_schedule_input = st.sidebar.text_input("Earn-Out Checkpoints (Months, comma-separated)", "24")\
earn_outs_schedule = [int(x.strip()) for x in earn_outs_schedule_input.split(',') if x.strip().isdigit()]\
\
earn_out_tiers = \{\}\
for checkpoint in earn_outs_schedule:\
    st.sidebar.markdown(f"**Checkpoint Month \{checkpoint\}**")\
    thresholds = st.sidebar.text_input(f"Attrition Thresholds for \{checkpoint\} (comma-separated)", "0.15,0.16,0.17,0.18,0.19")\
    multiples = st.sidebar.text_input(f"Multiples for \{checkpoint\} (comma-separated)", "5,4,3,2,1")\
    thresholds_list = [float(x.strip()) for x in thresholds.split(',')]\
    multiples_list = [float(x.strip()) for x in multiples.split(',')]\
    earn_out_tiers[checkpoint] = dict(zip(thresholds_list, multiples_list))\
\
# Front Book Inputs\
st.sidebar.subheader("Front Book")\
front_book_included = st.sidebar.radio("Include Front Book?", ["No", "Yes"]) == "Yes"\
front_book_volume = 0\
front_book_value_per_mid = 0\
front_book_years = 0\
if front_book_included:\
    front_book_volume = st.sidebar.number_input("New Adds per Month", value=100)\
    front_book_value_per_mid = st.sidebar.number_input("Net Contribution per MID ($)", value=30)\
    front_book_years = st.sidebar.number_input("New Adds Period (Years)", value=3)\
\
# Model Duration\
model_years = st.sidebar.selectbox("Model Duration (Years)", [5, 10])\
model_period_months = model_years * 12\
\
# --- Model Calculations ---\
monthly_attrition_rate = 1 - (1 - annual_attrition_rate) ** (1/12)\
initial_residual_base = portfolio_residual_monthly\
upfront_payment = -upfront_multiple * initial_residual_base\
\
# Set up DataFrames\
df = pd.DataFrame(index=np.arange(model_period_months))\
df['month'] = df.index + 1\
df['residual_base'] = 0.0\
df['cash_inflow_legacy'] = 0.0\
df['cash_inflow_front'] = 0.0\
\
residual_base = initial_residual_base\
front_residuals = np.zeros(model_period_months)\
current_attrition_rate = monthly_attrition_rate\
\
for idx in df.index:\
    month = df.loc[idx, 'month']\
    year = month // 12\
\
    if month > guarantee_period_months:\
        residual_base *= (1 - current_attrition_rate)\
\
    df.loc[idx, 'residual_base'] = residual_base\
    df.loc[idx, 'cash_inflow_legacy'] = residual_base\
\
    # Front book new adds logic\
    if front_book_included and month <= front_book_years * 12:\
        front_residuals[idx] += front_book_volume * front_book_value_per_mid\
\
    if idx > 0:\
        front_residuals[idx] += front_residuals[idx-1] * (1 - current_attrition_rate)\
\
    df.loc[idx, 'cash_inflow_front'] = front_residuals[idx]\
\
    # Step up attrition yearly\
    if month % 12 == 0:\
        current_attrition_rate += attrition_step_up_rate\
\
# --- Cash Flows and Earn-Outs ---\
\
cash_flows_no_front = [upfront_payment] + df['cash_inflow_legacy'].tolist()\
cash_flows_with_front = [upfront_payment] + (df['cash_inflow_legacy'] + df['cash_inflow_front']).tolist()\
\
# Earn-Outs (negative cash out)\
earn_out_payments = np.zeros(len(cash_flows_no_front))\
\
for checkpoint in earn_outs_schedule:\
    if checkpoint < len(df):\
        assumed_annual_attrition = annual_attrition_rate\
        tier_dict = earn_out_tiers.get(checkpoint, \{\})\
        best_match_multiple = 0\
        for threshold, multiple in sorted(tier_dict.items()):\
            if assumed_annual_attrition <= threshold:\
                best_match_multiple = multiple\
                break\
\
        if best_match_multiple > 0:\
            earn_out_payment = - (best_match_multiple * initial_residual_base)\
            earn_out_payments[checkpoint] = earn_out_payment\
\
cash_flows_no_front = np.array(cash_flows_no_front) + earn_out_payments\
cash_flows_with_front = np.array(cash_flows_with_front) + earn_out_payments\
\
# --- Metrics ---\
def calculate_metrics(cash_flows, gross_inflows):\
    cumulative_cf = np.cumsum(cash_flows)\
    total_invested = abs(upfront_payment) + abs(sum(earn_out_payments))\
    financial_moic = gross_inflows / total_invested\
    monthly_irr = npf.irr(cash_flows)\
    annualized_irr = (1 + monthly_irr) ** 12 - 1\
    breakeven_month = np.argmax(cumulative_cf >= 0)\
    return financial_moic, monthly_irr, annualized_irr, breakeven_month\
\
# Calculate separately\
metrics_no_front = calculate_metrics(cash_flows_no_front, sum(df['cash_inflow_legacy']))\
metrics_with_front = calculate_metrics(cash_flows_with_front, sum(df['cash_inflow_legacy'] + df['cash_inflow_front']))\
\
# --- Display Outputs ---\
st.header("\\ud83d\\udcca Model Outputs")\
\
st.subheader("Metrics WITHOUT Front Book")\
col1, col2, col3, col4 = st.columns(4)\
col1.metric("Financial MOIC", f"\{metrics_no_front[0]:.2f\}x")\
col2.metric("Annualized IRR", f"\{metrics_no_front[2]*100:.2f\}%")\
col3.metric("Monthly IRR", f"\{metrics_no_front[1]*100:.2f\}%")\
col4.metric("Breakeven Month", f"\{metrics_no_front[3]\}")\
\
if front_book_included:\
    st.subheader("Metrics WITH Front Book")\
    col5, col6, col7, col8 = st.columns(4)\
    col5.metric("Financial MOIC", f"\{metrics_with_front[0]:.2f\}x")\
    col6.metric("Annualized IRR", f"\{metrics_with_front[2]*100:.2f\}%")\
    col7.metric("Monthly IRR", f"\{metrics_with_front[1]*100:.2f\}%")\
    col8.metric("Breakeven Month", f"\{metrics_with_front[3]\}")\
\
# --- Cash Flow Plot ---\
st.subheader("Cumulative Cash Flow (No Front Book)")\
fig, ax = plt.subplots(figsize=(12, 6))\
ax.plot(df['month'], np.cumsum(cash_flows_no_front[1:]), label='Cumulative Cash Flow (No Front Book)')\
if front_book_included:\
    ax.plot(df['month'], np.cumsum(cash_flows_with_front[1:]), label='Cumulative Cash Flow (With Front Book)', linestyle='--')\
ax.axhline(0, color='gray', linestyle='--')\
ax.set_xlabel('Month')\
ax.set_ylabel('Cumulative Cash Flow ($)')\
ax.grid(True)\
ax.legend()\
st.pyplot(fig)\
\
# --- Sensitivity Table (No Front Book) ---\
st.subheader("\\ud83d\\udd22 Sensitivity Analysis (MOIC vs Attrition and Step-Up)")\
\
attrition_shifts = np.arange(-0.03, 0.04, 0.01)\
stepup_shifts = np.arange(0.00, 0.06, 0.01)\
\
sensitivity = pd.DataFrame()\
\
for stepup in stepup_shifts:\
    row = []\
    for shift in attrition_shifts:\
        monthly_attrition_adj = 1 - (1 - (annual_attrition_rate + shift))**(1/12)\
        residual = initial_residual_base\
        attrition_dynamic = monthly_attrition_adj\
        inflows = []\
\
        for month in range(model_period_months):\
            if month > guarantee_period_months:\
                residual *= (1 - attrition_dynamic)\
            inflows.append(residual)\
            if (month + 1) % 12 == 0:\
                attrition_dynamic += stepup\
\
        gross_inflows = sum(inflows)\
        financial_moic = gross_inflows / (abs(upfront_payment) + abs(sum(earn_out_payments)))\
        row.append(financial_moic)\
\
    sensitivity[f"Step-Up \{stepup*100:.0f\}%"] = row\
\
sensitivity.index = [f"Attr \{100*(annual_attrition_rate+shift):.1f\}%" for shift in attrition_shifts]\
st.dataframe(sensitivity.style.format("\{:.2f\}"))\
\
st.success("
\f1 \uc0\u9989 
\f0  Model Complete. Adjust parameters on the left and re-run!")\
}