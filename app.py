import streamlit as st


st.set_page_config(page_title="Policy Briefing Assistant", page_icon=":page_facing_up:")

st.title("Policy Briefing Assistant")

uploaded_file = st.file_uploader("Upload a file")
user_prompt = st.text_input("Enter your question or topic")

if st.button("Submit"):
    st.write("File uploaded:" if uploaded_file else "No file uploaded.")
    if uploaded_file:
        st.write(uploaded_file.name)

    st.write("Your input:" if user_prompt else "No text entered.")
    if user_prompt:
        st.write(user_prompt)
