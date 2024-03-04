import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() # Load all nevironment variables

from youtube_transcript_api import YouTubeTranscriptApi

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt="""You are an expert Yotube video summarizer. You will be taking the transcript text
and summarising the entire video and providing the important summary in points
within 300 to 600 words. Please provide the summary of the text given here:  """

def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e
    
# Getting the summary based on Prompt from Google Gemini Pro
def generate_gemini_summary(transcript_text,prompt):

    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt+transcript_text)
    return response.text

st.title("YouTube Transcript Summary Converter Powered by Google Gemini Pro 💁")
youtube_link = st.text_input("Enter YouTube Video Link:")

# Initialise session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if youtube_link:
    video_id = youtube_link.split("=")[1]
    print(video_id)
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Get Detailed Summary"):
    transcript_text=extract_transcript_details(youtube_link)

    if transcript_text:
        summary=generate_gemini_summary(transcript_text,prompt)
        st.markdown("## Detailed Summary:")
        st.write(summary)
        st.session_state['chat_history'].append(("You", youtube_link))
        st.session_state['chat_history'].append(("Gemini", summary))

st.subheader("Chat History:")
    
for role, text in st.session_state['chat_history']:
    if role == "Gemini":
        st.write(f"{role}:")
        st.write(text)
    else:
        st.write(f"{role}: {text}")