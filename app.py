import streamlit as st
import joblib
import requests
from sklearn.preprocessing import StandardScaler
import numpy as np
IMPORTING THE MODELS

scaler = joblib.load('scaler_model.joblib')
logistic_regression = joblib.load('./models/logistic_regression_model.joblib')
random_forest = joblib.load('./models/random_forest_model.joblib')
knn = joblib.load('./models/knn_model.joblib')

# # /////////////////////////////////////
# # PIPELINE FUNCTION TO PREPROCESS DATA
# # /////////////////////////////////////
scaler = StandardScaler()
def pipeline(data):
     predict_vals_2d = np.array(data).reshape(1, -1) # Reshape the data to a 2D array
     scaled_data = scaler.fit_transform(predict_vals_2d) # Scaler: Fit and transform the values
     return scaled_data

# # ////////////////////////
# # TITLE & TEAM INFO
# # ////////////////////////
st.title('Increase Credit Loan')



# # ////////////////////////
# # FORM
# # ////////////////////////



# ////////////////////////
# START -> LOADS & SHOWS THE MODELS
# ////////////////////////
st.title('ML models')
st.caption('We trained the following models:')
model_col1, model_col2 = st.columns(2, gap='large')
with model_col1:
     st.write('Linear Regression Classifier')
     st.write(logistic_regression)
with model_col2:
     st.write('K Nearest Neighbors Classifier')
     st.write(knn)
model_col3, model_col4 = st.columns(2, gap='large')
with model_col3:
     st.write('Random Forest Classifier')
     st.write(random_forest)
with model_col4:
     st.write('Standar Scaler model')
     st.write(scaler)
     

# st.title('See our results for yourself. Choose a model.')
# st.subheader('Click on "Submit" button after selecting a model & choosing an audio file.')
# with st.form('User_input'):
#     selected_model = st.selectbox('Model', ['Logistic_Regression_Classifier', 'K_Nearest_Neighbors_Classifier', 'Naive_Bayes_Classifier'])
#     wav_file = st.file_uploader('Select your own sound file')
#     if wav_file is not None:
#         uploaded_audio, _ = librosa.load(wav_file)
#     st.form_submit_button()

# if selected_model == "Logistic_Regression_Classifier":
#     if not wav_file:
#         st.subheader('No audio chosen.')
#     else:
#         user_audio = extract_time_domain_features(uploaded_audio)
#         log_reg_prediction = log_reg.predict([user_audio])
#         st.subheader('Result: {} species'.format(log_reg_prediction))
# elif selected_model == "K_Nearest_Neighbors_Classifier":
#     if not wav_file:
#         st.subheader('No audio chosen.')
#     else:
#         user_audio = extract_time_domain_features(uploaded_audio)
#         knn_prediction = knn.predict([user_audio])
#         st.subheader('Result: {} species'.format(knn_prediction))
# elif selected_model == "Naive_Bayes_Classifier":
#     if not wav_file:
#         st.subheader('No audio chosen.')
#     else:
#         user_audio = extract_time_domain_features(uploaded_audio)
#         naive_bayes_prediction = naive_bayes.predict([user_audio])
#         st.subheader('Result: {} species'.format(naive_bayes_prediction))










st.header(":mailbox: Get In Touch With Me!")


contact_form = """
<form action="https://formsubmit.co/YOUREMAIL@EMAIL.COM" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <button type="submit">Send</button>
</form>
"""

st.markdown(contact_form, unsafe_allow_html=True)

# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")