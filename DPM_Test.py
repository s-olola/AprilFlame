import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load('KNN_model.pkl')


# Add custom CSS for styling
st.markdown(
    """
    <style>
    /* Light blue background for the main content */
    .main {
        background-color: #d0e7ff;  /* Light blue background */
    }
    
    /* Style for input labels to make them bold */
    label {
        font-weight: bold;
    }

    /* Button styling */
    .stButton>button {
        font-weight: bold;
    }

    /* Prediction output styling */
    .prediction-output {
        font-size: 24px;
        font-weight: bold;
        color: black;  /* Black color for prediction output */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Title of the web-based interface
st.title("Dormancy Prediction Model")
st.header("Dormancy Status Detection")
st.caption("This returns the customer's dormancy status")

# Initialise a dictionary with all 181 features set to default values
default_features = {
    'NRGE': 0,
    'CALL_CNT': 0,
    'DATA_CNT': 0,
    'SMS_CNT': 0,
    'ACTIVE_CNT_CALL_SESSION': 0,
    'ACTIVE_CNT_CALL_DURATION': 0,
    'ACTIVE_CNT_RECHARGE': 0,
    'LAST_RECHARGE_DT_CNT': 0,
    'CUSTOMER_SEGMENT_KEY': 0,
    'TENURE_DAYS_CNT': 0,
    'TENURE_MONTHS_CNT': 0,
    'PACKAGE_KEY': 0,
    'YON': 0,
    'ACTIVE_CNT_RECHARGE_COUNT': 0,
    'ACTIVE_CNT_DATA_SPEND': 0,
    'TOTAL_NO_CC_CONTACT': 0,
    'ACTIVE_CNT_DATA_SESSION': 0,
    'ACTIVE_CNT_DATA_VOL': 0,
    'MIGRATION_CNT': 0,
    'DEVICE_TYPE_2G': False,
    'DEVICE_TYPE_3G': False,
    'DEVICE_TYPE_GSM': False,
    'DEVICE_TYPE_LTE': False,
    'DEVICE_TYPE_UNKNOWN': False,
    'TENUREX_02 1-12 months': False,
    'TENUREX_03 13-24 months': False,
    'TENUREX_04 25-48 months': False,
    'TENUREX_05 Above 4 years': False,
    'LAST_TX_TYPE_CALL_IN': False,
    'LAST_TX_TYPE_CALL_OUT': False,
    'LAST_TX_TYPE_DATA_USAGE': False,
    'LAST_TX_TYPE_SMS_SENT': False,
    'LAST_TX_TYPE_UNDEFINED': False,
    'LAST_TX_TYPE_VAS': False,
    'SERVICE_CLASS_BETATALK': False,
    'SERVICE_CLASS_BETATALK 250%': False,
    'SERVICE_CLASS_IPULSE': False,
    'SERVICE_CLASS_OTHER POSTPAID': False,
    'SERVICE_CLASS_OTHER PREPAID': False,
    'SERVICE_CLASS_PULSE': False,
    'SERVICE_CLASS_SME PLUS': False,
    'SERVICE_CLASS_SMOOTHTALK': False,
    'SERVICE_CLASS_STARTER PACK': False,
    'SERVICE_CLASS_SUPERSAVER': False,
    'SERVICE_CLASS_TRUTALK': False,
    'SERVICE_CLASS_XTRA PRO': False,
    'SERVICE_CLASS_XTRA SPECIAL': False,
    'SERVICE_CLASS_XTRA SPECIAL POSTPAID': False,
    'SERVICE_CLASS_ZONE': False,
    'GEOREGION_NORTH CENTRAL': False,
    'GEOREGION_NORTH EAST': False,
    'GEOREGION_NORTH WEST': False,
    'GEOREGION_SOUTH EAST': False,
    'GEOREGION_SOUTH SOUTH': False,
    'GEOREGION_SOUTH WEST': False,
    'NGSTATE_ABIA': False,
    'NGSTATE_ADAMAWA': False,
    'NGSTATE_AKWA IBOM': False,
    'NGSTATE_ANAMBRA': False,
    'NGSTATE_BAUCHI': False,
    'NGSTATE_BAYELSA': False,
    'NGSTATE_BENUE': False,
    'NGSTATE_BORNO': False,
    'NGSTATE_CROSS RIVER': False,
    'NGSTATE_DELTA': False,
    'NGSTATE_EBONYI': False,
    'NGSTATE_EDO': False,
    'NGSTATE_EKITI': False,
    'NGSTATE_ENUGU': False,
    'NGSTATE_FCT': False,
    'NGSTATE_GOMBE': False,
    'NGSTATE_IMO': False,
    'NGSTATE_JIGAWA': False,
    'NGSTATE_KADUNA': False,
    'NGSTATE_KANO': False,
    'NGSTATE_KATSINA': False,
    'NGSTATE_KEBBI': False,
    'NGSTATE_KOGI': False,
    'NGSTATE_KWARA': False,
    'NGSTATE_LAGOS': False,
    'NGSTATE_NASSARAWA': False,
    'NGSTATE_NIGER': False,
    'NGSTATE_OGUN': False,
    'NGSTATE_ONDO': False,
    'NGSTATE_OSUN': False,
    'NGSTATE_OYO': False,
    'NGSTATE_PLATEAU': False,
    'NGSTATE_RIVERS': False,
    'NGSTATE_SOKOTO': False,
    'NGSTATE_TARABA': False,
    'NGSTATE_YOBE': False,
    'NGSTATE_ZAMFARA': False,
    'TERRITORY_ABA': False,
    'TERRITORY_ABEOKUTA': False,
    'TERRITORY_ABUJA CITY': False,
    'TERRITORY_ADO EKITI': False,
    'TERRITORY_AGEGE': False,
    'TERRITORY_ALIMOSHO': False,
    'TERRITORY_APAPA': False,
    'TERRITORY_ASABA': False,
    'TERRITORY_AWKA': False,
    'TERRITORY_BAUCHI': False,
    'TERRITORY_BIU': False,
    'TERRITORY_CALABAR': False,
    'TERRITORY_DAMATURU': False,
    'TERRITORY_DUTSE': False,
    'TERRITORY_EBONYI': False,
    'TERRITORY_ENUGU': False,
    'TERRITORY_ETSAKO': False,
    'TERRITORY_FAGGE': False,
    'TERRITORY_FESTAC': False,
    'TERRITORY_FUNTUA': False,
    'TERRITORY_GBOKO': False,
    'TERRITORY_GOMBE': False,
    'TERRITORY_GUSAU': False,
    'TERRITORY_GWAGWALADA': False,
    'TERRITORY_IBADAN': False,
    'TERRITORY_IFE CENTRAL': False,
    'TERRITORY_IJEBU-ODE': False,
    'TERRITORY_IKEJA': False,
    'TERRITORY_IKORODU': False,
    'TERRITORY_IKPOBA-OKHA': False,
    'TERRITORY_ILORIN': False,
    'TERRITORY_JALINGO': False,
    'TERRITORY_JOS': False,
    'TERRITORY_KADUNA NORTH': False,
    'TERRITORY_KADUNA SOUTH': False,
    'TERRITORY_KAFANCHAN': False,
    'TERRITORY_KANO CENTRAL': False,
    'TERRITORY_KATAGUM': False,
    'TERRITORY_KATSINA': False,
    'TERRITORY_KEBBI': False,
    'TERRITORY_KEFFI': False,
    'TERRITORY_KONTAGORA': False,
    'TERRITORY_KOSOFE': False,
    'TERRITORY_KUBWA': False,
    'TERRITORY_LAFIA': False,
    'TERRITORY_LAGOS ISLAND': False,
    'TERRITORY_LAGOS MAINLAND': False,
    'TERRITORY_LEKKI': False,
    'TERRITORY_LOKOJA': False,
    'TERRITORY_MAIDUGURI': False,
    'TERRITORY_MAKURDI': False,
    'TERRITORY_MINNA': False,
    'TERRITORY_MUBI': False,
    'TERRITORY_MUSHIN': False,
    'TERRITORY_NNEWI': False,
    'TERRITORY_OBIO/AKPOR 1': False,
    'TERRITORY_OBIO/AKPOR 2': False,
    'TERRITORY_OGBOMOSHO': False,
    'TERRITORY_OKENE': False,
    'TERRITORY_ONDO': False,
    'TERRITORY_ONITSHA': False,
    'TERRITORY_OREDO': False,
    'TERRITORY_ORLU': False,
    'TERRITORY_OSOGBO': False,
    'TERRITORY_OWERRI 1': False,
    'TERRITORY_OWERRI 2': False,
    'TERRITORY_OYIGBO': False,
    'TERRITORY_PORT HARCOURT': False,
    'TERRITORY_SAKI': False,
    'TERRITORY_SANGO': False,
    'TERRITORY_SOKOTO': False,
    'TERRITORY_UGHELLI': False,
    'TERRITORY_UMUAHIA': False,
    'TERRITORY_UYO': False,
    'TERRITORY_WARRI': False,
    'TERRITORY_WUKARI': False,
    'TERRITORY_YENAGOA': False,
    'TERRITORY_YOLA': False,
    'TERRITORY_ZARIA': False,
    'DATA_USAGE_Data User': False,
    'DATA_USAGE_Non Data': False,
    'SMS_USAGE_Non SMS': False,
    'SMS_USAGE_SMS User': False,
    'VOICE_USAGE_Non-Voice': False,
    'VOICE_USAGE_Voice User': False,
    'SERVICE_CNT_01_No Service': False,
    'SERVICE_CNT_02_One Service': False,
    'SERVICE_CNT_03_Two Services': False,
    'SERVICE_CNT_04_Three Services': False,
}

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    # Input fields for the first column
    nrge = st.number_input('Number of Recharge Events (NRGE)', min_value=0, max_value=100, value=default_features['NRGE'], step=1)
    call_cnt = st.number_input('Number of Calls (CALL_CNT)', min_value=0, max_value=100, value=default_features['CALL_CNT'], step=1)
    data_cnt = st.number_input('Number of Data Sessions (DATA_CNT)', min_value=0, max_value=100, value=default_features['DATA_CNT'], step=1)
    sms_cnt = st.number_input('Number of SMS Sent (SMS_CNT)', min_value=0, max_value=100, value=default_features['SMS_CNT'], step=1)
    data_user = st.selectbox('Data User', options=[True, False])

with col2:
    # Input fields for the second column
    customer_segment_key = st.number_input('Customer Segment Key (CUSTOMER_SEGMENT_KEY)', min_value=0, max_value=100, value=default_features['CUSTOMER_SEGMENT_KEY'], step=1)
    tenure_days_cnt = st.number_input('Tenure Days Count (TENURE_DAYS_CNT)', min_value=0, max_value=5000, value=default_features['TENURE_DAYS_CNT'], step=1)
    last_recharge_dt_cnt = st.number_input('Last Recharge Days Count (LAST_RECHARGE_DT_CNT)', min_value=0, max_value=5000, value=default_features['LAST_RECHARGE_DT_CNT'], step=1)
    voice_user = st.selectbox('Voice User', options=[True, False])
    sms_user = st.selectbox('SMS User', options=[True, False])  


# Update the dictionary with user inputs
default_features.update({
    'NRGE': nrge,
    'CALL_CNT': call_cnt,
    'DATA_CNT': data_cnt,
    'SMS_CNT': sms_cnt,
    'CUSTOMER_SEGMENT_KEY': customer_segment_key,
    'TENURE_DAYS_CNT': tenure_days_cnt,
    'LAST_RECHARGE_DT_CNT': last_recharge_dt_cnt,
    'DATA_USAGE_Data User': data_user,
    'VOICE_USAGE_Voice User': voice_user,
})

# Convert the dictionary to a DataFrame to ensure correct format for prediction
full_feature_vector = pd.DataFrame([default_features])

# Predict button
if st.button('Predict'):
    # Make a prediction using the full feature vector
    prediction = model.predict(full_feature_vector)
    
    # Interpret prediction
    if prediction[0] == 0:
        prediction_text = "Active"
    else:
        prediction_text = "Dormant"
    
    # Display the prediction with styled output
    st.markdown(f"<div class='prediction-output'>The predicted class is: <strong>{prediction_text}</strong></div>", unsafe_allow_html=True)

 