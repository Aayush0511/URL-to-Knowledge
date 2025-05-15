import os
import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.schema import Document
from dotenv import load_dotenv
import yt_dlp
import requests
import pyttsx3
import base64
import tempfile

# Initialize session state
if "summary" not in st.session_state:
    st.session_state.summary = None
if "docs" not in st.session_state:
    st.session_state.docs = None
if "llm" not in st.session_state:
    st.session_state.llm = None

def get_transcript_yt_dlp(url):
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'no_warnings': True
    }

    transcript_text = ""
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        captions = info.get("subtitles") or info.get("automatic_captions")

        transcript_url = None
        if captions:
            lang = list(captions.keys())[0]
            transcript_url = captions[lang][0]['url']

        if transcript_url:
            response = requests.get(transcript_url)
            if "WEBVTT" in response.text[:10]:
                lines = []
                for line in response.text.splitlines():
                    if line.strip() == '' or '-->' in line or line.lower().startswith("webvtt"):
                        continue
                    lines.append(line.strip())
                transcript_text = ' '.join(lines)

        if not transcript_text:
            transcript_text = info.get("description", "")

    return transcript_text

def save_summary_to_file(summary_text):
    filename = "summary.txt"
    with open(filename, "w", encoding='utf-8') as f:
        f.write(summary_text)
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Download Summary</a>'
    return href

def text_to_speech(summary_text):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', int(rate * 0.8))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        temp_filename = tmp_file.name

    engine.save_to_file(summary_text, temp_filename)
    engine.runAndWait()
    engine.stop()

    with open(temp_filename, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return audio_bytes

load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="LangChain: Summarize & QA From URL", page_icon="🦜")
st.title("🦜 LangChain: Summarize & QA From YouTube or Website")
st.subheader('Summarize URL Content + Ask Questions!')

# Sidebar
with st.sidebar:
    groq_api_key = os.getenv("GROQ_API_KEY")

    word_limit = st.number_input("Summary length (in words)", min_value=100, value=500, step=50)

    model_options = [
        "gemma2-9b-it",
        "llama3-8b-8192",
        "llama3-70b-8192",
    ]
    selected_model = st.selectbox("Select GROQ Model", model_options)

# URL input
generic_url = st.text_input("Paste YouTube or Website URL", label_visibility="collapsed")

# Summary generation
if st.button("Summarize"):
    if not groq_api_key.strip():
        st.error("Please provide a valid GROQ API key.")
    elif not generic_url.strip() or not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Extracting and summarizing..."):
                if "youtube.com" in generic_url:
                    transcript = get_transcript_yt_dlp(generic_url)
                    docs = [Document(page_content=transcript)]
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                llm = ChatGroq(model=selected_model, groq_api_key=groq_api_key)

                prompt_template = f"""
                Provide a summary of the following content in approximately {word_limit} words:
                Content: {{text}}
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.session_state.summary = output_summary
                st.session_state.docs = docs
                st.session_state.llm = llm

                st.success("✅ Summary Generated")
        except Exception as e:
            st.exception(f"Exception: {e}")

# Display summary if exists
if st.session_state.summary:
    st.subheader("📝 Summary")
    st.success(st.session_state.summary)

    # Download summary
    st.markdown(save_summary_to_file(st.session_state.summary), unsafe_allow_html=True)

    # Audio output
    st.audio(text_to_speech(st.session_state.summary), format='audio/mp3')

    # Ask Questions Section
    st.markdown("### 🤖 Ask a Question Based on the Content")
    user_question = st.text_input("Ask your question here...")
    if st.button("Ask"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                qa_prompt_template = """
                Based on the following content, answer the user's question.

                Content:
                {content}

                Question:
                {question}

                Answer:
                """
                full_prompt = qa_prompt_template.format(
                    content=st.session_state.docs[0].page_content,
                    question=user_question
                )
                try:
                    answer = st.session_state.llm.predict(full_prompt)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Error answering question: {e}")
