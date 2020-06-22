import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

# st.title('My first app')
# st.write("Here's our first attempt at using data to create a table:")
# st.write(pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# }))


df = pd.DataFrame({
  'first column': [1, 2, 3, 4 , 5],
  'second column': [10, 20, 30, 40, 50]
})

df

a = 'str_arun'
a


# import time 

# 'Starting a long computation...'

# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.1)

'...and now we\'re done!'


uploaded_file = st.file_uploader("Choose a CSV file", type="py")
if uploaded_file is not None:
    st.write("successful")


import streamlit as st
import os

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)

import streamlit as st
import os

filename = st.text_input('Enter a file path:')
try:
    with open(filename) as input:
        st.text(input.read())
except FileNotFoundError:
    st.error('File not found.')
    

map_data = pd.DataFrame(
    [[52.2, -0.1275], [51.3,0.143]],
    columns=['lat', 'lon'])

st.map(map_data)    
    
    
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'

st.markdown(get_table_download_link(data), unsafe_allow_html=True)