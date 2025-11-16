import os
import io
import json
import re
import hashlib
from datetime import datetime
import streamlit as st
from gtts import gTTS
from google import genai
from google.genai import types
import plotly.graph_objects as go

# ============================
# CONFIG
# ============================
TEXT_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are a senior interviewer.
Transcribe candidate audio, then ask the next question.
Return ONLY JSON:

{
  "transcript": "latest candidate answer",
  "ai_reply": "next interviewer question"
}
"""

ANALYSIS_PROMPT = """
You are an expert interview evaluator.

Return ONLY JSON:
{
  "total_score": 0,
  "communication_score": 0,
  "communication_level": "",
  "sentiment": "",
  "topics_covered": [],
  "mistakes": [],
  "correct_answers": [],
  "feedback": ""
}
"""

# ============================
# JSON HELPERS
# ============================
def extract_json(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    blob = match.group(0).replace("\n", " ").replace("\t", " ")
    blob = re.sub(r",\s*}", "}", blob)
    blob = re.sub(r",\s*]", "]", blob)
    try:
        return json.loads(blob)
    except:
        try:
            blob = re.sub(r"(\w+):", r'"\1":', blob)
            return json.loads(blob)
        except:
            return None

def safe_analysis_json(raw):
    defaults = {
        "total_score": 70,
        "communication_score": 70,
        "communication_level": "Average",
        "sentiment": "Neutral",
        "topics_covered": ["General Discussion"],
        "mistakes": ["No mistakes provided by AI"],
        "correct_answers": ["Candidate demonstrated some understanding."],
        "feedback": "Good attempt. Try improving clarity and answer depth."
    }
    data = extract_json(raw)
    if not data:
        return defaults
    for key, val in defaults.items():
        if key not in data or data[key] in ["", None, [], {}]:
            data[key] = val
    for key in ["total_score", "communication_score"]:
        try:
            data[key] = float(data[key])
        except:
            data[key] = 0
    return data

# ============================
# HELPERS
# ============================
def get_client(api_key):
    return genai.Client(api_key=api_key)

def history_to_text(history):
    output = []
    q_num = 1
    a_num = 1
    for m in history:
        if m["role"] == "ai":
            output.append(f"Q{q_num}: {m['text']}")
            q_num += 1
        else:
            output.append(f"A{a_num}: {m['text']}")
            a_num += 1
    return "\n".join(output)

def gtts_cached(text):
    os.makedirs("tts_cache", exist_ok=True)
    fn = "tts_cache/" + hashlib.md5(text.encode()).hexdigest() + ".mp3"
    if not os.path.exists(fn):
        tts = gTTS(text)
        tts.save(fn)
    return fn

def interview_turn(client, audio_bytes, history):
    prompt = f"{SYSTEM_PROMPT}\n\nConversation:\n{history_to_text(history)}"
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt, types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")],
        config=types.GenerateContentConfig(response_mime_type="text/plain", temperature=0.3, max_output_tokens=600)
    )
    raw = response.text
    data = extract_json(raw)
    if not data:
        return "[Transcription failed]", "Could you repeat that?"
    return data.get("transcript", ""), data.get("ai_reply", "")

def ai_interview_analysis(client, history, tone):
    transcript = history_to_text(history)
    prompt = f"{ANALYSIS_PROMPT}\n\nTone: {tone}\n\nFull Transcript:\n{transcript}"
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2, max_output_tokens=1000)
    )
    raw = response.text
    return safe_analysis_json(raw)

# ============================
# STREAMLIT APP SETUP
# ============================
st.set_page_config(page_title="AI Voice & Webcam Interview", layout="wide")
st.title("💼 AI Voice & Webcam Interview Application")
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# Initialize session state
for key in ["started","history","finished","tone","current_question","transcript",
            "interview_type","candidate_name","candidate_experience","resume_file"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key in ["current_question","transcript","tone","interview_type","candidate_name","candidate_experience"] else [] if key=="history" else False

# ============================
# PAGE NAVIGATION
# ============================
page = st.sidebar.radio("Navigate Pages", ["Voice Interview", "📊 Interview Analysis"])

st.sidebar.write("This API Key : 'AIzaSyBl79ucqcFpALPspGWtftOxeecqpg8t3ZA'")

# ============================
# PRE-INTERVIEW SETUP
# ============================

if not st.session_state.started and page=="Voice Interview":
    st.subheader("📝 Requirements Details")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.candidate_name = st.text_input("Candidate Name")
            st.session_state.interview_type = st.selectbox(
                "Interview Domain", 
                ["General", "Technical", "HR/Behavioral", "Managerial", "Group Discussion"]
            )
            st.session_state.candidate_experience = st.selectbox(
                "Candidate Experience Level", 
                ["Fresher", "Experienced"]
            )
            st.session_state.interview_level = st.selectbox(
                "Interview Level",
                ["Beginner", "Intermediate", "Advanced"]
            )
        with col2:
            st.session_state.tone = st.selectbox(
                "Feedback Tone", 
                ["Professional", "Encouraging", "Friendly", "Critical"]
            )
            st.session_state.resume_file = st.file_uploader(
                "Upload your Resume (PDF/DOCX)", 
                type=["pdf","docx"]
            )

    if st.button("🚀 Start Interview"):
        if not api_key:
            st.warning("Enter Gemini API key first.")
        elif not st.session_state.candidate_name:
            st.warning("Enter candidate name.")
        else:
            intro_text = f"Hello {st.session_state.candidate_name}, thank you for coming in today. " \
                         f"I'll be your AI interviewer for this {st.session_state.interview_type} interview " \
                         f"({st.session_state.interview_level} level). Let's begin. Could you please tell me a bit about yourself?"
            st.session_state.started = True
            st.session_state.history = [{"role":"ai","text": intro_text}]
            st.session_state.current_question = intro_text
            st.rerun()
    st.stop()

# ============================
# CLIENT
# ============================
if api_key:
    client = get_client(api_key)
else:
    client = None

# ============================
# VOICE INTERVIEW PAGE
# ============================
if page=="Voice Interview":
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("🤝 Conversation")
        chat_container = st.container()
        for idx, m in enumerate(st.session_state.history):
            is_ai = m['role'] == 'ai'
            timestamp = datetime.now().strftime("%H:%M")
            col_left, col_right = st.columns([3,1]) if is_ai else st.columns([1,3])
            with col_left if is_ai else col_right:
                bubble_color = "#d9f7ff" if is_ai else "#fff0d6"
                st.markdown(f"<div style='background:{bubble_color}; padding:10px; border-radius:12px; margin-bottom:5px;'>"
                            f"<b>{'AI' if is_ai else 'You'}</b> <span style='font-size:10px; color:gray'>{timestamp}</span><br>"
                            f"{m['text']}</div>", unsafe_allow_html=True)
                if is_ai:
                    st.audio(gtts_cached(m['text']))
                elif 'audio_bytes' in m:
                    st.audio(m['audio_bytes'])

    with col2:
        st.subheader("📷 Web Controls")
        st.camera_input("Enable your camera (optional)")
        if st.session_state.current_question:
            st.markdown(f"**Current Question:** {st.session_state.current_question}")
            st.audio(gtts_cached(st.session_state.current_question))

        audio_file = st.audio_input("Record your answer")
        col_send, col_finish = st.columns(2)
        with col_send:
            if st.button("📤 Send") and audio_file and client:
                # ===== FIXED READ ISSUE =====
                audio_file_bytes = audio_file.read()  # read once
                transcript, reply = interview_turn(client, audio_file_bytes, st.session_state.history)
                st.session_state.history.append({"role":"user","text": transcript, "audio_bytes": audio_file_bytes})
                st.session_state.history.append({"role":"ai","text": reply})
                st.session_state.current_question = reply
                st.rerun()
        with col_finish:
            if st.button("🏁 End  Interview"):
                st.session_state.finished = True
                st.rerun()

# ============================
# INTERVIEW ANALYSIS PAGE
# ============================
if page=="📊 Interview Analysis" and st.session_state.finished and client:
    st.header("📊 Interview Analysis")
    analysis = ai_interview_analysis(client, st.session_state.history, st.session_state.tone)

    tabs = st.tabs(["Scores","Feedback","Charts","Voice Transcript Replay","Insights & Suggestions"])

    with tabs[0]:
        st.subheader("🧮 Scores")
        colA, colB, colC = st.columns(3)
        colA.metric("Total Score", analysis["total_score"])
        colB.metric("Communication Score", analysis["communication_score"])
        colC.metric("Communication Level", analysis["communication_level"])

    with tabs[1]:
        st.subheader("📝 Feedback & Details")
        with st.expander("Feedback"):
            st.write(analysis["feedback"])
        with st.expander("Mistakes"):
            st.write(", ".join(analysis["mistakes"]))
        with st.expander("Correct Answers"):
            st.write(", ".join(analysis["correct_answers"]))
        with st.expander("Topics Covered"):
            st.write(", ".join(analysis["topics_covered"]))

    with tabs[2]:
        st.subheader("📈 Charts")
        confidence = float(analysis.get("total_score",0))
        clarity = float(analysis.get("communication_score",0))
        empathy = 100.0
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_conf = go.Figure(go.Indicator(mode="gauge+number", value=confidence, title={'text':'Confidence'}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':'#0b8457'}}))
            st.plotly_chart(fig_conf, use_container_width=True)
        with col2:
            fig_clarity = go.Figure(go.Indicator(mode="gauge+number", value=clarity, title={'text':'Clarity'}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':'#f4b400'}}))
            st.plotly_chart(fig_clarity, use_container_width=True)
        with col3:
            fig_empathy = go.Figure(go.Indicator(mode="gauge+number", value=empathy, title={'text':'Empathy'}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':'#db4437'}}))
            st.plotly_chart(fig_empathy, use_container_width=True)

    with tabs[3]:
        st.subheader("🔊 Voice Transcript Replay")
        for m in st.session_state.history:
            is_ai = m['role']=='ai'
            st.markdown(f"**{'AI' if is_ai else 'Candidate'}:** {m['text']}")
            if is_ai:
                st.audio(gtts_cached(m['text']))
            elif 'audio_bytes' in m:
                st.audio(m['audio_bytes'])

    with tabs[4]:
        st.subheader("💡 Insights & Suggestions")
        for idx, m in enumerate(st.session_state.history):
            if m['role']=='user':
                st.markdown(f"**Q{idx+1}:** {st.session_state.history[idx-1]['text'] if idx>0 else ''}")
                st.markdown(f"**Candidate Answer:** {m['text']}")
                if 'audio_bytes' in m:
                    st.audio(m['audio_bytes'])
                st.markdown(f"**Insight:** Shows strengths and areas for improvement.")

    pdf_content = "=== INTERVIEW TRANSCRIPT ===\n\n"
    for m in st.session_state.history:
        role = "AI" if m['role']=='ai' else "Candidate"
        pdf_content += f"{role}: {m['text']}\n"
    pdf_content += "\n\n=== ANALYSIS ===\n" + json.dumps(analysis, indent=2)
    st.download_button("📄 Download Report", pdf_content, "interview_analysis.txt")
