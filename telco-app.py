import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import numpy as np
import plotly.express as px

# Train the model
model = LinearRegression()

data = pd.read_excel('Telco_customer_churn_adapted_v2.xlsx')

# Encode categorical features 
device_class_mapping = {'Low End': 1, 'Mid End': 2, 'High End': 3}
data['Device Class'] = data['Device Class'].map(device_class_mapping)

payment_method_mapping = {'Digital Wallet': 1, 'Pulsa': 2, 'Debit': 3, 'Credit': 4}  
data['Payment Method'] = data['Payment Method'].map(payment_method_mapping)

def handle_no_internet_service(col):
  return col.apply(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else 2)

data['Games Product'] = handle_no_internet_service(data['Games Product']) 
data['Music Product'] = handle_no_internet_service(data['Music Product'])
data['Education Product'] = handle_no_internet_service(data['Education Product'])  
data['Video Product'] = handle_no_internet_service(data['Video Product'])
data['Call Center'] = handle_no_internet_service(data['Call Center']) 
data['Use MyApp'] = handle_no_internet_service(data['Use MyApp'])

X = data[['Tenure Months', 'Monthly Purchase (Thou. IDR)',  
          'Games Product', 'Music Product', 'Education Product', 
          'Video Product', 'Call Center', 'Use MyApp']]

y = data['CLTV (Predicted Thou. IDR)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

model.fit(X_train, y_train)

# Streamlit app
st.title('Customer Churn Prediction App')

if st.sidebar.selectbox("Choose Page", ["Description", "Predict CLTV"]) == "Description":
    st.markdown('''
    This app predicts the Customer Lifetime Value (CLTV) for a telecom customer based on their characteristics.

    Use the sidebar to enter details and get a prediction!
    ''')

    # Charts
    churn_pct = data['Churn Label'].value_counts(normalize=True) * 100

    fig_churn = px.pie(churn_pct, names=churn_pct.index, title='% Churn')
    st.plotly_chart(fig_churn)

    fig_churn_device = px.histogram(data.groupby(['Device Class', 'Churn Label'])['Tenure Months'].count(),
                  barmode='group', title='Churn by Device')
    st.plotly_chart(fig_churn_device)

else:
    st.sidebar.subheader('Enter Customer Details')
    tenure = st.sidebar.number_input('Tenure Months', min_value=0, max_value=100, value=12)
    monthly_purchase = st.sidebar.number_input('Monthly Purchase (Thou. IDR)', min_value=0.0, max_value=1000.0, value=10.0)

    games_product = st.sidebar.radio('Games Product', ['Yes', 'No', 'No internet service'])
    music_product = st.sidebar.radio('Music Product', ['Yes', 'No', 'No internet service'])
    education_product = st.sidebar.radio('Education Product', ['Yes', 'No', 'No internet service'])
    video_product = st.sidebar.radio('Video Product', ['Yes', 'No', 'No internet service'])

    call_center = st.sidebar.radio('Call Center', ['Yes', 'No'])
    use_myapp = st.sidebar.radio('Use MyApp', ['Yes', 'No'])

    games_product = 1 if games_product == 'Yes' else 0 if games_product == 'No' else 2
    music_product = 1 if music_product == 'Yes' else 0 if music_product == 'No' else 2
    education_product = 1 if education_product == 'Yes' else 0 if education_product == 'No' else 2
    video_product = 1 if video_product == 'Yes' else 0 if video_product == 'No' else 2
    call_center = 1 if call_center == 'Yes' else 0
    use_myapp = 1 if use_myapp == 'Yes' else 0

    X_new = pd.DataFrame([[tenure, monthly_purchase, games_product, music_product, 
                         education_product, video_product, call_center, use_myapp]],
                       columns=X.columns)

    if st.sidebar.button('Predict'):
        prediction = model.predict(X_new)[0]

        st.subheader('Prediction Results')
        st.write(f'The predicted CLTV is {prediction:.2f} Thou. IDR')
