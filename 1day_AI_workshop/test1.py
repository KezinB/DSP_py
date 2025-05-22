import streamlit as st

# Set page title
st.set_page_config(page_title="Simple Streamlit UI", layout="centered")

# App Title
st.title("🚀 Simple Streamlit UI Example")

# Text input
name = st.text_input("Enter your name:")

# Number input
age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)

# Dropdown select box
color = st.selectbox("Choose your favorite color:", ["Red", "Blue", "Green", "Yellow"])

# Checkbox
subscribe = st.checkbox("Subscribe to newsletter")

# Button
if st.button("Submit"):
    st.success("Form Submitted Successfully!")
    st.write(f"👤 Name: {name}")
    st.write(f"🎂 Age: {age}")
    st.write(f"🎨 Favorite Color: {color}")
    st.write(f"📬 Subscribed: {'Yes' if subscribe else 'No'}")
