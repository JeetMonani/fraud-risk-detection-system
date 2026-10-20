import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detector", layout="centered")
st.title("🛡️ Live Fraud Risk Assessment Dashboard")

st.write("Adjust transaction parameters to compute live ML fraud scores.")

# --- CATEGORICAL INPUTS ---
st.subheader("🛒 Transaction Context")
col1, col2 = st.columns(2)

with col1:
    payment_method = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Bank Transfer", "Debit Card"])
    customer_location = st.selectbox("Customer Location", ["United States", "United Kingdom", "Asia", "Europe"])

with col2:
    product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Home & Kitchen", "Health & Beauty"])
    device_used = st.selectbox("Device Used", ["Mobile", "Desktop", "Tablet"])

address_mismatch = st.selectbox("Address Mismatch? (Shipping != Billing)", ["No", "Yes"])
mismatch_val = 1 if address_mismatch == "Yes" else 0

# --- NUMERICAL INPUTS ---
st.subheader("👤 Customer & Order Details")
col3, col4 = st.columns(2)

with col3:
    transaction_amount = st.number_input("Transaction Amount ($)", min_value=1.0, value=150.0)
    quantity = st.number_input("Quantity of Items", min_value=1, max_value=100, value=1)
    customer_age = st.slider("Customer Age", 18, 100, 30)

with col4:
    account_age_days = st.number_input("Account Age (Days)", min_value=0, max_value=10000, value=365)
    transaction_hour = st.slider("Transaction Hour (24h format)", 0, 23, 14)

# --- PREDICTION TRIGGER ---
st.markdown("---")
if st.button("Analyze Transaction", use_container_width=True):
    # This payload map now matches your exact 10 model features flawlessly
    payload = {
        "Payment Method": payment_method,
        "Product Category": product_category,
        "Customer Location": customer_location,
        "Device Used": device_used,
        "Address_Mismatch": mismatch_val,
        "Transaction Amount": transaction_amount, 
        "Quantity": quantity,
        "Customer Age": customer_age,
        "Account Age Days": account_age_days,
        "Transaction Hour": transaction_hour
    }
    
    # Fire request to backend
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()
        
        if "error" in result:
            st.error(f"Backend Processing Error: {result['error']}")
        else:
            prob = result["fraud_probability"]
            
            # Display results beautifully based on risk level
            if prob > 70:
                st.error(f"🚨 **CRITICAL RISK** | Fraud Probability: {prob}%")
                st.warning("Action Recommended: Decline transaction automatically and flag user account.")
            elif prob > 40:
                st.warning(f"⚠️ **MEDIUM RISK** | Fraud Probability: {prob}%")
                st.info("Action Recommended: Trigger secondary Multi-Factor Authentication (MFA).")
            else:
                st.success(f"✅ **LOW RISK** | Fraud Probability: {prob}%")
                st.caption("Action Recommended: Approve transaction instantly.")
                
    except Exception as e:
        st.error(f"Could not connect to FastAPI server. Is it running? Error: {e}")