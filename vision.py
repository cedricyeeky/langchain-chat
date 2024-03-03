import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() 

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,image,prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input,image[0],prompt])
    return response.text
    

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


# Initialise streamlit app
st.set_page_config(page_title="Invoice/FS LLM Image Renderer")

# Initialise session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.header("Invoice/Financial Statement LLM Image Renderer using Google Gemini Pro 💁")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image=""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)


submit = st.button("Tell me about the image")

input_prompt = """
               You are a field expert in understanding invoices and financial statements.
               You will receive input invoices and financial statements in the form of uploaded images &
               you will have to answer questions based on the input image
               """

# If submit button is clicked
if submit:
    image_data = input_image_setup(uploaded_file)
    response=get_gemini_response(input_prompt,image_data,input)
    st.subheader("Gemini:")
    st.write(response)
    st.session_state['chat_history'].append(("You", input))
    st.session_state['chat_history'].append(("Gemini", response))

st.subheader("Chat History:")
    
for role, text in st.session_state['chat_history']:
    if role == "Gemini":
        st.write(f"{role}:")
        st.write(text)
    else:
        st.write(f"{role}: {text}")

