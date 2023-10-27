import pickle
import streamlit as st

model = pickle.load(open('estimasi_apple.sav', 'rb'))

st.title('Apple Stock High Price')

close = st.number_input('Closing Price')
low = st.number_input('Lowest Price')
open = st.number_input('Opening Price of the day')
volume = st.number_input('Volume of stock traded')
adjClose = st.number_input('Closing stock price in relation')
adjLow = st.number_input('Lowest stock price')
adjOpen = st.number_input('Opening stock price')
adjVolume = st.number_input('Trading volume')

predict = ''

if st.button('Submit'):
    predict = model.predict(
        [[close, low, open, volume, adjClose, adjLow, adjOpen, adjVolume]]
    )
    st.write('Apple Stock High Price:', predict[0])
