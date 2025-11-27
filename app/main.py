import streamlit as st
from prediction_helper import predict
import pandas as pd  # Needed for a cleaner display of inputs
from report_generator import generate_risk_report_pdf
from io import BytesIO

# --- 1. CONFIGURATION AND STYLING ---
# Set the page configuration for a professional look
st.set_page_config(
    page_title="Lauki Finance: Credit Risk Modelling",
    page_icon="🏦",
    layout="wide"  # Use the full width of the browser
)

# Use Markdown for a stylized main title
st.markdown(
    """
    <style>
    .big-font {
        font-size:36px !important;
        font-weight: bold;
        color: #1E90FF; /* Dodger Blue for branding */
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<p class="big-font">Lauki Finance: Automated Credit Risk Decision Tool</p>', unsafe_allow_html=True)
st.write("---")

# --- 2. INPUT GATHERING (Organized into Sections) ---

# Section 1: Applicant and Loan Details
st.header("👤 Applicant & Loan Profile")
col_A, col_B, col_C = st.columns(3)

with col_A:
    age = st.number_input('Age (Years)', min_value=18, step=1, max_value=100, value=28)
    # FIX: Changed label to remove '₹'
    income = st.number_input('Annual Income (Currency Unit)', min_value=0, value=1200000,
                             help="The applicant's gross yearly income.")

with col_B:
    # FIX: Changed label to remove '₹'
    loan_amount = st.number_input('Requested Loan Amount (Currency Unit)', min_value=0, value=2560000,
                                  help="Total amount requested by the applicant.")
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36,
                                         help="Requested repayment period.")

with col_C:
    # Calculated Feature - Display only
    loan_to_income_ratio = loan_amount / income if income > 0 else 0
    st.markdown("#### Loan-to-Income Ratio")
    st.info(f"{loan_to_income_ratio:.2f}", icon="📈")

    # Categorical Inputs
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])

st.write("---")

# Section 2: Credit History Metrics
st.header("📜 Credit History Metrics")
col_D, col_E, col_F = st.columns(3)

with col_D:
    avg_dpd_per_delinquency = st.number_input('Avg DPD (Days Past Due)', min_value=0, value=20,
                                              help="Average delay in payments for past delinquencies.")
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=10, step=1, value=2,
                                        help="Number of active loans/credit lines.")

with col_E:
    delinquency_ratio = st.number_input('Delinquency Ratio (%)', min_value=0, max_value=100, step=1, value=30,
                                        help="Percentage of accounts with past due status.")

with col_F:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio (%)', min_value=0, max_value=100, step=1,
                                               value=30, help="Percentage of total available credit being used.")
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])

st.write("---")

# --- 3. EXECUTION AND RESULTS ---

# Place the button in the main flow
if st.button('🎯 Calculate Risk Score & Decision', type="primary"):

    # 1. Prediction Call
    probability, credit_score, rating = predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type)

    # 2. Results Container
    # Use a container for styling the results block
    with st.container():
        st.subheader("✅ Credit Decision Summary")

        # Determine color for the rating display
        if rating == 'Excellent':
            rating_color = '#4CAF50'  # Green
        elif rating == 'Poor':
            rating_color = '#F44336'  # Red
        else:
            rating_color = '#FF9800'  # Orange (Fair)

        # Display key metrics using metrics/columns for impact
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(label="Default Probability", value=f"{probability:.2%}", delta="Risk Exposure")

        with res_col2:
            st.metric(label="Calculated Credit Score", value=f"{credit_score}")

        with res_col3:
            st.markdown(f"#### Rating")
            st.markdown(
                f'<div style="background-color: {rating_color}; color: white; padding: 10px; border-radius: 5px; text-align: center;"><h1>{rating}</h1></div>',
                unsafe_allow_html=True)

    st.write("---")

    # 3. Data Collection for Report
    # FIX: Ensure dictionary keys and string values do not contain '₹'
    input_data_for_report = {
        'Age (Years)': str(age),
        'Annual Income (Unit)': f"{income:,}",  # Formatted string
        'Requested Loan Amount (Unit)': f"{loan_amount:,}",  # Formatted string
        'Loan Tenure (months)': str(loan_tenure_months),
        'Loan-to-Income Ratio': f"{loan_to_income_ratio:.3f}",
        'Avg DPD': str(avg_dpd_per_delinquency),
        'Delinquency Ratio (%)': str(delinquency_ratio),
        'Credit Utilization Ratio (%)': str(credit_utilization_ratio),
        'Open Loan Accounts': str(num_open_accounts),
        'Residence Type': residence_type,
        'Loan Purpose': loan_purpose,
        'Loan Type': loan_type
    }

    results_for_report = {
        'probability': probability,
        'credit_score': credit_score,
        'rating': rating
    }

    # 4. Input Review and Download Button
    st.subheader("🔍 Input Data for Review & Audit")

    # Display Inputs in a table
    df_inputs = pd.DataFrame(input_data_for_report.items(), columns=['Feature', 'Value'])
    st.dataframe(df_inputs.set_index('Feature'), width="stretch")

    # Add the PDF Download Button
    # The pdf_bytes variable now receives bytes directly from pdf.output(dest='S')
    pdf_bytes = generate_risk_report_pdf(input_data_for_report, results_for_report)

    st.download_button(
        label="⬇️ Download PDF Risk Report",
        data=pdf_bytes,
        file_name=f"Lauki_Credit_Report_{age}_{loan_amount}.pdf",
        mime="application/pdf",
        type="secondary"
    )

# --- Footer ---
st.caption("Powered by ML • Built by Ruchi • Codebasics ML Bootcamp Project - Optimized for Credit Risk Management.")