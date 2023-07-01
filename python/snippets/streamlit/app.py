import streamlit as st
from PIL import Image

st.set_page_config(page_title="Dashboard", page_icon=":tada:", layout="wide") # To select a different emoji for the page_icon, use https://webfx.com/tools/emoji-cheat-sheet/
# We can also use the default layout which is center by not using the layout parameter

# Load assets
img_cable = Image.open("images/CDAD.png")

# Header section
with st.container():
    st.subheader("Hello World!")
    st.title("Welcome!")
    st.write("This is plain text for your website")

with st.container():
    st.write("---") # Divider line
    left_column, rigth_column = st.columns((1, 2))
    with left_column:
        st.header("Next section")
        st.write("##") # Simply a spacer
        st.write("This is the text for the next section")
    
    with rigth_column:
        st.header("Right side")
        st.write("Right side")
        st.image(img_cable)