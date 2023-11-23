import time
import streamlit as st
import joblib
import requests
import numpy as np
from streamlit_lottie import st_lottie

# # ////////////////////////
# IMPORTING THE MODELS
# # ////////////////////////
scaler = joblib.load('./models/scaler_model.joblib')
logistic_regression = joblib.load('./models/logistic_regression_model.joblib')
random_forest = joblib.load('./models/random_forest_model.joblib')
knn = joblib.load('./models/knn_model.joblib')

# # /////////////////////////////////////
# # PIPELINE FUNCTION TO PREPROCESS DATA
# # /////////////////////////////////////


def pipeline(data):
    predict_vals_2d = np.array(data).reshape(
        1, -1)  # Reshape the data to a 2D array
    # Scaler: Fit and transform the values
    scaled_data = scaler.transform(predict_vals_2d)
    return scaled_data

# # /////////////////////////////////////
# # GLOBAL VARIABLES
# # /////////////////////////////////////
# variable_map = {
#     'age': 0,
#     'income': 0,
#     'month': 0,
#     'num_bank_accounts': 0,
#     'num_credit_card': 0,
#     'last_interest_rate': 0,
#     'Num_of_Loan': 0,
#     'month_delay': 0,
#     'payments_delay': 0,
#     'credit_mix': 0,
#     'Credit_History_Age': 0,
#     'Monthly_Balance': 0
# }


st.title('Increase Your Credit Loan')

# # ////////////////////////
# # FORM
# # ////////////////////////
# with st.form("credit_form"):
#    st.write("Fill in your data")
#    #    my_number = st.slider('Pick a number', 1, 10)
#    my_color = st.selectbox('Pick a color', ['red','orange','green','blue','violet'])
#    st.form_submit_button('Submit my picks')

# - What is your age? -> Age
# - What is your annual income? -> Annual Income
# - Enter the month of your credit solicitude -> Month
# - How many Bank Accounts do you have? -> Num_Bank_Accounts
# - How many Credit Cards do you have?  -> Num_Credit_Card
# - What was your last interest rate? -> Interest_Rate
# - How many loans have you had in the past? -> Num_of_Loan
# - What is the maximum delay you have had? -> Delay_from_due_date
# - How many payments have you been delayed? -> Num_of_Delayed_payment
# - Credit_Mix
# - Credit_History_Age
# - What is your last monthly balance? -> Monthly_Balance


# # ////////////////////////
# # START -> LOADS & SHOWS THE MODELS
# # ////////////////////////
# st.title('ML models')
# st.caption('We trained the following models:')
# model_col1, model_col2 = st.columns(2, gap='large')
# with model_col1:
#      st.write('Linear Regression Classifier')
#      st.write(logistic_regression)
# with model_col2:
#      st.write('K Nearest Neighbors Classifier')
#      st.write(knn)
# model_col3, model_col4 = st.columns(2, gap='large')
# with model_col3:
#      st.write('Random Forest Classifier')
#      st.write(random_forest)
# with model_col4:
#      st.write('Standar Scaler model')
#      st.write(scaler)


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


# st.header(":mailbox: Get In Touch With Me!")


# contact_form = """
# <form action="https://formsubmit.co/YOUREMAIL@EMAIL.COM" method="POST">
#      <input type="hidden" name="_captcha" value="false">
#      <input type="text" name="name" placeholder="Your name" required>
#      <input type="email" name="email" placeholder="Your email" required>
#      <textarea name="message" placeholder="Your message here"></textarea>
#      <button type="submit">Send</button>
# </form>
# """

# st.markdown(contact_form, unsafe_allow_html=True)

# # Use Local CSS File
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# local_css("style/style.css")


################################################################################################################################################################################################################################################
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 1
if 'displayName' not in st.session_state:
    st.session_state['displayName'] = ''
if 'age' not in st.session_state:
    st.session_state['age'] = 0
if 'income' not in st.session_state:
    st.session_state['income'] = 0
if 'month' not in st.session_state:
    st.session_state['month'] = 0
if 'num_bank_accounts' not in st.session_state:
    st.session_state['num_bank_accounts'] = 0
if 'num_credit_card' not in st.session_state:
    st.session_state['num_credit_card'] = 0
if 'last_interest_rate' not in st.session_state:
    st.session_state['last_interest_rate'] = 0
if 'Num_of_Loan' not in st.session_state:
    st.session_state['Num_of_Loan'] = 0
if 'month_delay' not in st.session_state:
    st.session_state['month_delay'] = 0
if 'payments_delay' not in st.session_state:
    st.session_state['payments_delay'] = 0
if 'credit_mix' not in st.session_state:
    st.session_state['credit_mix'] = 0
if 'Credit_History_Age' not in st.session_state:
    st.session_state['Credit_History_Age'] = 0
if 'Monthly_Balance' not in st.session_state:
    st.session_state['Monthly_Balance'] = 0


def update_age():
    st.session_state.age = st.session_state._age


def update_income():
    st.session_state.income = st.session_state._income


def update_month():
    if st.session_state._month == 'January':
        st.session_state.month = '1'
    elif st.session_state._month == 'February':
        st.session_state.month = '2'
    elif st.session_state._month == 'March':
        st.session_state.month = '3'
    elif st.session_state._month == 'April':
        st.session_state.month = '4'
    elif st.session_state._month == 'May':
        st.session_state.month = '5'
    elif st.session_state._month == 'June':
        st.session_state.month = '6'
    elif st.session_state._month == 'July':
        st.session_state.month = '7'
    elif st.session_state._month == 'August':
        st.session_state.month = '8'
    elif st.session_state._month == 'September':
        st.session_state.month = '9'
    elif st.session_state._month == 'October':
        st.session_state.month = '10'
    elif st.session_state._month == 'November':
        st.session_state.month = '11'
    elif st.session_state._month == 'December':
        st.session_state.month = '12'


def update_bank_accounts():
    st.session_state.num_bank_accounts = st.session_state._num_bank_accounts


def update_credit_card():
    st.session_state.num_credit_card = st.session_state._num_credit_card


def update_last_interest_rate():
    st.session_state.last_interest_rate = st.session_state._last_interest_rate


def update_Num_of_Loan():
    st.session_state.Num_of_Loan = st.session_state._Num_of_Loan


def update_month_delay():
    st.session_state.month_delay = st.session_state._month_delay


def update_payments_delay():
    st.session_state.payments_delay = st.session_state._payments_delay


def update_Credit_History_Age():
    st.session_state.Credit_History_Age = st.session_state._Credit_History_Age


def update_payments_delay():
    st.session_state.payments_delay = st.session_state._payments_delay


def update_Monthly_Balance():
    st.session_state.Monthly_Balance = st.session_state._Monthly_Balance


def update_credit_mix():
    if st.session_state._credit_mix == 'Good':
        st.session_state.credit_mix = '2'
    elif st.session_state._credit_mix == 'Standard':
        st.session_state.credit_mix = '1'
    elif st.session_state._credit_mix == 'Bad':
        st.session_state.credit_mix = '0'


def simulate_load_snowflake_table():
    success = False
    response_message = ''
    num_rows = 0

    try:
        success = True
        response_message = 'Success'
        num_rows = 2
    except Exception as e:
        response_message = e

    return success, response_message, num_rows


def set_page_view(page):
    st.session_state['current_step'] = 1
#     st.session_state['queued_file'] = None
#     st.session_state['current_view'] = page


def update_displayName():
    st.session_state.displayName = st.session_state._displayName


def set_form_step(action, step=None):
    if action == 'Next':
        st.session_state['current_step'] = st.session_state['current_step'] + 1
    if action == 'Back':
        st.session_state['current_step'] = st.session_state['current_step'] - 1
    if action == 'Jump':
        st.session_state['current_step'] = step

##### wizard functions ####


def wizard_form_header():
    st.caption('Fill the input and click next.')
    st.text_input(label='Type the same user of the chatbot:',
                  value=st.session_state.displayName, key='_displayName', on_change=update_displayName)

    # determines button color which should be red when user is on that given step
    age_type = 'primary' if st.session_state['current_step'] == 1 else 'secondary'
    income_type = 'primary' if st.session_state['current_step'] == 2 else 'secondary'
    month_type = 'primary' if st.session_state['current_step'] == 3 else 'secondary'
    bank_accounts_type = 'primary' if st.session_state['current_step'] == 4 else 'secondary'

    credit_cards_type = 'primary' if st.session_state['current_step'] == 5 else 'secondary'
    interest_rate_type = 'primary' if st.session_state['current_step'] == 6 else 'secondary'
    number_loans_type = 'primary' if st.session_state['current_step'] == 7 else 'secondary'
    max_delay_type = 'primary' if st.session_state['current_step'] == 8 else 'secondary'

    num_delayed_payments_type = 'primary' if st.session_state['current_step'] == 9 else 'secondary'
    credit_mix_type = 'primary' if st.session_state['current_step'] == 10 else 'secondary'
    credit_history_type = 'primary' if st.session_state['current_step'] == 11 else 'secondary'
    monthly_balance_type = 'primary' if st.session_state['current_step'] == 12 else 'secondary'

    step_cols1 = st.columns([.5, .85, .85, .85, .85, .5])
    step_cols2 = st.columns([.5, .85, .85, .85, .85, .5])
    step_cols3 = st.columns([.5, .85, .85, .85, .85, .5])

    step_cols1[2].button('\tAge', on_click=set_form_step,
                         args=['Jump', 1], type=age_type)
    step_cols1[3].button('\tIncome', on_click=set_form_step,
                         args=['Jump', 2], type=income_type)
    step_cols1[1].button('\tMonth', on_click=set_form_step,
                         args=['Jump', 3], type=month_type)
    step_cols1[4].button('Bank Accounts', on_click=set_form_step, args=[
                         'Jump', 4], type=bank_accounts_type)

    step_cols2[1].button('Credit Cards', on_click=set_form_step, args=[
                         'Jump', 5], type=credit_cards_type)
    step_cols2[2].button('Interest Rate', on_click=set_form_step, args=[
                         'Jump', 6], type=interest_rate_type)
    step_cols2[3].button('Number of Loans', on_click=set_form_step, args=[
                         'Jump', 7], type=number_loans_type)
    step_cols2[4].button('Maximum Delay', on_click=set_form_step, args=[
                         'Jump', 8], type=max_delay_type)

    step_cols3[1].button('Number of Delayed Payments', on_click=set_form_step, args=[
                         'Jump', 9], type=num_delayed_payments_type)
    step_cols3[2].button('Credit Mix', on_click=set_form_step, args=[
                         'Jump', 10], type=credit_mix_type)
    step_cols3[3].button('Credit History Age', on_click=set_form_step, args=[
                         'Jump', 11], type=credit_history_type)
    step_cols3[4].button('Monthly Balance', on_click=set_form_step, args=[
                         'Jump', 12], type=monthly_balance_type)


### Replace Wizard Form Body with this ###
def wizard_form_body():

    ###### Step 1: Month ######
    if st.session_state['current_step'] == 3:
        st.markdown('\n')
        st.markdown('\n')
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
        st.selectbox('Month of credit solicitude:', key='_month',
                     options=months, index=0, on_change=update_month)

    ###### Step 2: Age ######
    if st.session_state['current_step'] == 1:
        st.markdown('\n')
        st.markdown('\n')
    #    age = st.number_input('Type your age', value = age, on_change = update_age)
        st.number_input('Type your age', key='_age',
                        value=st.session_state.age, on_change=update_age)

    ###### Step 3: Income ######
    if st.session_state['current_step'] == 2:
        st.markdown('\n')
        st.markdown('\n')
        st.number_input('Type your monthly income', key='_income',
                        value=st.session_state.income, on_change=update_income)

    ###### Step 4: Bank Accounts ######
    if st.session_state['current_step'] == 4:
        st.markdown('\n')
        st.markdown('\n')
        st.number_input('Type the number of distinct bank accounts that you have:', key='_num_bank_accounts',
                        value=st.session_state.num_bank_accounts, on_change=update_bank_accounts)

    ###### Step 5: Credit Cards ######
    if st.session_state['current_step'] == 5:
        st.markdown('\n')
        st.markdown('\n')
        st.number_input('Type the number of distinct credit cards that you have:', key='_num_credit_card',
                        value=st.session_state.num_credit_card, on_change=update_credit_card)

    ###### Step 6: Interest Rate ######
    if st.session_state['current_step'] == 6:
        st.markdown('\n')
        st.markdown('\n')
        st.number_input('Type the last interest rate that you paid:', key='_last_interest_rate',
                        value=st.session_state.last_interest_rate, on_change=update_last_interest_rate)

    ###### Step 7: Number of loans ######
    if st.session_state['current_step'] == 7:
        st.markdown('\n')
        st.markdown('\n')
        st.number_input('Type how many loans have you had in the past:', key='_Num_of_Loan',
                        value=st.session_state.Num_of_Loan, on_change=update_Num_of_Loan)

    ###### Step 8: Maximum month delay ######
    if st.session_state['current_step'] == 8:
        st.markdown('\n')
        st.markdown('\n')
        st.number_input('Type the maximum month delay you have had from payment due date:',
                        key='_month_delay', value=st.session_state.month_delay, on_change=update_month_delay)

    ###### Step 9: Maximum payments delay ######
    if st.session_state['current_step'] == 9:
        st.markdown('\n')
        st.markdown('\n')
        st.number_input('Type the maximum payments delay you have had:', key='_payments_delay',
                        value=st.session_state.payments_delay, on_change=update_payments_delay)

    ###### Step 10: Credit Mix ######
    if st.session_state['current_step'] == 10:
        st.markdown('\n')
        st.markdown('\n')
        credit_mix_options = ['Good', 'Standard', 'Bad']
        st.selectbox('What is your credit mix status?', options=credit_mix_options,
                     index=0, key='_credit_mix', on_change=update_credit_mix)

    ###### Step 11: Credit History Age ######
    if st.session_state['current_step'] == 11:
        st.markdown('\n')
        st.markdown('\n')
        st.number_input('Type the number of years you have had a credit history:', key='_Credit_History_Age',
                        value=st.session_state.Credit_History_Age, on_change=update_Credit_History_Age)

    ###### Step 12: Monthly Balance ######
    if st.session_state['current_step'] == 11:
        st.markdown('\n')
        st.markdown('\n')
        Monthly_Balance = st.number_input('Type your last monthly balance:', key='_Monthly_Balance',
                                          value=st.session_state.Monthly_Balance, on_change=update_Monthly_Balance)

    st.markdown('---')

    form_footer_container = st.empty()
    with form_footer_container.container():

        disable_back_button = True if st.session_state['current_step'] == 1 else False
        disable_next_button = True if st.session_state['current_step'] == 12 else False

        form_footer_cols = st.columns([5, 1, 1, 1.75])

        form_footer_cols[1].button('Back', on_click=set_form_step, args=[
                                   'Back'], disabled=disable_back_button)
        form_footer_cols[2].button('Next', on_click=set_form_step, args=[
                                   'Next', ], disabled=disable_next_button)

        submit_ready = False if st.session_state['current_step'] == 12 != None else True
        submit = form_footer_cols[3].button('📤 Submit', disabled=submit_ready)

    if submit:
        #    [9, 28, 34847.84, 2, 4, 6, 10, 3.0, 1.0, 27.250000,485.298,2]
        data = pipeline([
            float(st.session_state.month),
            float(st.session_state.age),
            float(st.session_state.income),
            float(st.session_state.num_bank_accounts),
            float(st.session_state.num_credit_card),
            float(st.session_state.last_interest_rate),
            float(st.session_state.Num_of_Loan),
            float(st.session_state.month_delay),
            float(st.session_state.payments_delay),
            float(st.session_state.Credit_History_Age),
            float(st.session_state.Monthly_Balance),
            float(st.session_state.credit_mix),
        ])
        print(data)
        result = knn.predict(data)
        json_data = {
            "displayName": st.session_state.displayName,
            "creditResult": True if result == 1 else False
        }

        url = 'http://158.101.21.48:3000/user/credit'
        response = requests.post(url, json=json_data)

        print("0 -> no | 1 -> yes: ", result)
        with st.spinner('Loading your data to the model ...'):
            time.sleep(3)
            success, response_message, num_rows = simulate_load_snowflake_table()
            if success:
                if result == 1:
                    st.success(
                        f'✅Congratulations we\'ve approved your credit. Please return to the chatbot.')
                else:
                    st.error(
                        f'❌ Unfortunately we\'ve declined your credit.  Please return to the chatbot.')
            else:
                st.error(f'❌  Failed to load data to the model. Please try again.')

            ok_cols = st.columns(8)
            ok_cols[0].button('OK', type='primary', on_click=set_page_view, args=[
                              'Grid'], use_container_width=True)


def render_wizard_view():
    with st.expander('Fill your data for our credit model', expanded=True):
        wizard_form_header()
        wizard_form_body()


render_wizard_view()
