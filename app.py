# app.py — AI HOME TUTOR + STORY AGENT + Crew-like Helper (Groq-backed)
from dotenv import load_dotenv
load_dotenv()

import os
import io
import traceback
import streamlit as st
from groq import Groq

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from xml.sax.saxutils import escape

# -------------------------
# GROQ CLIENT
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY not set in .env — Groq calls will fail.")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def groq_chat(prompt: str, model="llama-3.1-8b-instant", temperature=0.5, max_retries=2):
    """Robust wrapper for Groq chat completions."""
    if not client:
        return "Groq client not configured. Set GROQ_API_KEY in .env."
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            # robust extraction
            choice = getattr(resp, "choices", None)
            if choice:
                # attempt to fetch message.content
                first = choice[0]
                message = getattr(first, "message", None)
                if message:
                    # message may be object-like or dict-like
                    content = getattr(message, "content", None)
                    if content:
                        return content
                    if isinstance(message, dict):
                        return message.get("content", str(resp))
                # fallback
                for attr in ("text", "content"):
                    if hasattr(first, attr):
                        return getattr(first, attr)
            # last fallback
            return str(resp)
        except Exception as e:
            if attempt < max_retries:
                import time
                time.sleep(0.8 + attempt * 0.5)
                continue
            return f"Groq request failed: {e}\n{traceback.format_exc()}"

# -------------------------
# AI TUTOR ENGINE
# -------------------------
def ask_tutor(prompt):
    system = (
        "You are a friendly, safe AI home tutor for kids aged 6–14. "
        "Explain concepts simply using short sentences, examples, and emojis. "
        "Keep tone positive and age-appropriate. Include a 2–3 question mini-quiz at the end."
    )
    full_prompt = f"{system}\n\nUser request:\n{prompt}"
    return groq_chat(full_prompt, temperature=0.5)

# -------------------------
# STORY AGENT ENGINE
# -------------------------
def generate_story(prompt):
    system = (
        "You are a kids' story writer. Write safe, fun, simple, moral-based stories for ages 4–12. "
        "Use emojis, short paragraphs, friendly dialogue, and end with a clear moral."
    )
    full_prompt = f"{system}\n\nStory request:\n{prompt}"
    return groq_chat(full_prompt, temperature=0.8)

# -------------------------
# CREW-LIKE HELPER (Groq-backed)
# -------------------------
def crew_ai_helper_using_groq(user_query: str, role_hint: str = None):
    """
    Simulates a lightweight 'Crew' helper by dispatching a structured prompt to Groq.
    Keeps things simple and avoids the crewa i package / LiteLLM dependency.
    """
    system = "You are a helpful multi-step assistant (Crew-like). Break down the task, give step-by-step guidance, and provide a short actionable result."
    if role_hint:
        system += f" You should act as: {role_hint}."
    prompt = f"{system}\n\nUser request:\n{user_query}\n\nReturn a clear step-by-step plan and a short concise answer."
    return groq_chat(prompt, temperature=0.45)

# -------------------------
# PDF GENERATOR
# -------------------------
def generate_pdf(title, body):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20*mm,
        rightMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm
    )

    title_style = ParagraphStyle(
        "title",
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#4a90e2"),
        spaceAfter=12,
    )
    body_style = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=12,
        leading=16,
    )

    story = [
        Paragraph(escape(title), title_style),
        Spacer(1, 10),
        Paragraph(escape(body).replace("\n", "<br/>"), body_style),
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="AI Home Tutor", page_icon="🎓", layout="centered")
st.title("🎓 AI Home Tutor For Kids")
st.markdown("🌈 Fun learning + magical stories + Crew-like helper (Groq)")

# Tabs
tab1, tab2, tab3 = st.tabs(["📚 Tutor Mode", "📖 Story Agent", "🤖 Crew Helper"])

# 1: Tutor Mode
with tab1:
    st.header("📘 Ask Your Tutor Anything")
    topic = st.text_input("What do you want to learn today?", key="topic_input")
    difficulty = st.selectbox(
        "Choose Difficulty",
        ["Beginner (Age 6–8)", "Intermediate (Age 9–11)", "Advanced (Age 12–14)"],
        key="diff_select"
    )

    if st.button("✨ Generate Lesson", key="gen_lesson"):
        if not topic or not topic.strip():
            st.error("Please enter a topic.")
        else:
            prompt = (
                f"Topic: {topic}\nDifficulty: {difficulty}\n"
                "Explain step-by-step with emojis, 1–2 examples, a short fun quiz (2 questions), and learning tips."
            )
            lesson = ask_tutor(prompt)
            st.session_state["lesson"] = lesson
            st.success("Lesson Ready!")

    if st.session_state.get("lesson"):
        st.subheader("📘 Your Lesson")
        edited_lesson = st.text_area("Edit lesson (optional)", st.session_state["lesson"], height=300)
        pdf_bytes = generate_pdf(topic or "Lesson", edited_lesson)
        st.download_button("📄 Download Lesson PDF", data=pdf_bytes, file_name="lesson.pdf", mime="application/pdf")

# 2: Story Agent
with tab2:
    st.header("📖 Create a Fun Story!")
    story_type = st.selectbox(
        "Choose Story Type",
        ["Moral Story", "Adventure", "Animal Story", "Bedtime Story", "Fantasy Story"],
        key="story_type"
    )
    story_topic = st.text_input("Story Topic (e.g., honesty, a brave rabbit, magical forest)", key="story_topic")

    if st.button("🌟 Generate Story", key="gen_story"):
        if not story_topic or not story_topic.strip():
            st.error("Please enter a topic.")
        else:
            prompt = f"Write a {story_type.lower()} for kids about: {story_topic}. Use emojis, simple language, dialogues, and end with a moral."
            story = generate_story(prompt)
            st.session_state["story"] = story
            st.success("Story Generated!")

    if st.session_state.get("story"):
        st.subheader("📖 Your Story")
        edited_story = st.text_area("Edit story (optional)", st.session_state["story"], height=300)
        pdf_bytes = generate_pdf(f"{story_type} - {story_topic}", edited_story)
        st.download_button("📄 Download Story PDF", data=pdf_bytes, file_name="story.pdf", mime="application/pdf")

# 3: Crew-like Helper
with tab3:
    st.header("🤖 Crew-like Helper (Groq-backed)")
    query = st.text_area("Ask the Crew-like helper (ideas, lesson-plan steps, creative prompts, breakdowns):", key="crew_query")
    role_hint = st.text_input("Optional role hint (e.g., 'lesson planner', 'quiz maker')", key="crew_role")
    if st.button("🚀 Ask Helper", key="crew_ask"):
        if not query or not query.strip():
            st.error("Please enter a question.")
        else:
            try:
                reply = crew_ai_helper_using_groq(query, role_hint=role_hint.strip() or None)
                st.subheader("🧠 Helper Response")
                st.write(reply)
            except Exception as e:
                st.error("Helper failed:")
                st.code(str(e))

# Footer
st.markdown(
    "<p style='text-align:center; color:gray; margin-top:35px;'>"
    "Developed by <b>Vaishnavi Bhosale</b> 🌸"
    "</p>",
    unsafe_allow_html=True
)
