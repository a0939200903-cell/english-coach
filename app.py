from gtts import gTTS
import io
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
import time
import edge_tts
from google import genai
from google.genai import types
from pydantic import BaseModel
import requests
import streamlit as st

st.set_page_config(
    page_title="AI 英文隨身教練",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

PROFILE_FILE = "student_profile.json"
TEXT_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
]


# --- Pydantic 資料結構 ---
class VocabItem(BaseModel):
    word: str
    meaning_zh: str
    example: str


class DialogueTurn(BaseModel):
    speaker: str  # 只能是 "Alex" 或 "Maya"
    text: str


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_index: int
    explanation_zh: str


class DailyLesson(BaseModel):
    topic_title: str
    category: str
    article: str
    vocabulary: list[VocabItem]
    podcast_script: list[DialogueTurn]
    comprehension_quiz: list[QuizQuestion]
    speaking_questions: list[str]


class QuestionFeedback(BaseModel):
    transcript_or_summary_zh: str
    grammar_and_naturalness_zh: str
    casual_native_rewrite: str
    toeic_850_rewrite: str


class SpeakingEvaluation(BaseModel):
    feedbacks: list[QuestionFeedback]
    score_delta: int
    updated_weaknesses: list[str]
    next_difficulty_note: str
    ai_diagnostic_reason_zh: str
    overall_encouragement_zh: str


# --- 讀取設定與金鑰 ---
def get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


def get_gemini_client():
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "⚠️ 尚未設定 GEMINI_API_KEY！請在 Streamlit Cloud 的 Secrets 中填入。"
        )
        st.stop()
    return genai.Client(api_key=api_key)


def score_to_cefr(score: int) -> str:
    if score < 550:
        return "A2"
    elif score < 785:
        return "B1"
    elif score < 945:
        return "B2"
    else:
        return "C1"


# --- GitHub 雲端進度同步（確保 Streamlit 重啟後進度不遺失） ---
def default_profile() -> dict:
    return {
        "estimated_toeic": 550,
        "cefr_level": "B1",
        "strengths": ["Business Vocabulary"],
        "weaknesses": [
            "Daily Conversational English",
            "Advanced Grammar",
        ],
        "difficulty_note": "循序漸進提升口語自然度與多益商務理解",
        "lessons_log": [],
        "vocab_bank": [],
        "history": ["建立雲端學習檔案（起始預估多益 550 分）"],
    }


def load_profile() -> dict:
    if "profile" in st.session_state:
        return st.session_state["profile"]

    gh_token = get_secret("GITHUB_TOKEN")
    gh_repo = get_secret("GITHUB_REPO")

    if gh_token and gh_repo:
        url = (
            f"https://api.github.com/repos/{gh_repo}/contents/{PROFILE_FILE}"
        )
        headers = {
            "Authorization": f"token {gh_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                content_b64 = resp.json()["content"]
                data = json.loads(
                    base64.b64decode(content_b64).decode("utf-8")
                )
                st.session_state["profile"] = data
                return data
        except Exception:
            pass

    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            st.session_state["profile"] = data
            return data

    data = default_profile()
    st.session_state["profile"] = data
    return data


def save_profile(profile: dict):
    st.session_state["profile"] = profile
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    gh_token = get_secret("GITHUB_TOKEN")
    gh_repo = get_secret("GITHUB_REPO")
    if gh_token and gh_repo:
        url = (
            f"https://api.github.com/repos/{gh_repo}/contents/{PROFILE_FILE}"
        )
        headers = {
            "Authorization": f"token {gh_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        try:
            sha = None
            get_resp = requests.get(url, headers=headers, timeout=10)
            if get_resp.status_code == 200:
                sha = get_resp.json().get("sha")

            encoded = base64.b64encode(
                json.dumps(profile, ensure_ascii=False, indent=2).encode("utf-8")
            ).decode("utf-8")
            payload = {
                "message": f"Update learning progress ({datetime.date.today()})",
                "content": encoded,
            }
            if sha:
                payload["sha"] = sha
            requests.put(url, headers=headers, json=payload, timeout=10)
        except Exception as e:
            st.warning(f"雲端同步暫時延遲（本機進度已保留）：{e}")


# --- Gemini 呼叫與自然語音合成 ---
def call_gemini_structured(contents, schema, temperature: float = 0.7):
    client = get_gemini_client()
    for _ in range(2):
        for model_name in TEXT_MODELS:
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=temperature,
                    ),
                )
                return json.loads(resp.text)
            except Exception:
                time.sleep(1)
    raise RuntimeError("目前 AI 伺服器尖峰忙碌中，請稍候 10 秒再按一次！")


def synthesize_podcast_mp3(script: list[dict]) -> bytes:
    try:
        async def _gen():
            voices = {
                "Alex": "en-US-AndrewNeural",
                "Maya": "en-US-AvaNeural",
            }
            audio_chunks = bytearray()
            for turn in script:
                voice = voices.get(turn["speaker"], "en-US-AvaNeural")
                comm = edge_tts.Communicate(turn["text"], voice)
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        audio_chunks.extend(chunk["data"])
            if not audio_chunks:
                raise RuntimeError("Empty audio received from edge_tts")
            return bytes(audio_chunks)

        def _thread_runner():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_gen())
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_thread_runner).result()
    except Exception:
        mp3_fp = io.BytesIO()
        for turn in script:
            text = turn.get("text", "").strip()
            if not text:
                continue
            accent = "com" if turn.get("speaker") == "Alex" else "co.uk"
            tts = gTTS(text=text, lang="en", tld=accent)
            tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.read()


# --- 密碼保護鎖（防止陌生人耗用你的 API 額度） ---
app_pwd = get_secret("APP_PASSWORD")
if app_pwd and not st.session_state.get("authenticated"):
    st.title("🔒 專屬 AI 英文教練")
    pwd_input = st.text_input("請輸入個人通行密碼：", type="password")
    if st.button("進入教室", use_container_width=True):
        if pwd_input == app_pwd:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("密碼錯誤，請再試一次")
    st.stop()


# --- 主畫面介面 ---
profile = load_profile()
today_str = datetime.date.today().strftime("%Y-%m-%d")

st.markdown(
    f"### 🎧 隨身英文特訓 &nbsp; `多益預估 {profile['estimated_toeic']} 分 ({profile.get('cefr_level', 'B1')})`"
)

tab1, tab2, tab3 = st.tabs(["📖 今日課程", "🗣️ 口說實戰", "📊 學習儀表板"])

# ==================== 分頁 1：今日課程 ====================
with tab1:
    latest_lesson = profile.get("latest_lesson")
    is_today = profile.get("latest_lesson_date") == today_str

    btn_label = (
        "🔄 重新生成今日課程" if is_today else "✨ 生成今日專屬課程與 Podcast"
    )
    if st.button(btn_label, type="primary", use_container_width=True):
        with st.spinner("🧠 正在為你量身打造文章、單字與雙人 Podcast..."):
            prompt = f"""
            你是一位能動態調整教學難度的專屬英文家教。
            【學生檔案】
- 目前預估多益分數：{profile['estimated_toeic']} 分（目標 800 分以上，並能流暢與外國人日常聊天）
- CEFR 等級：{profile.get('cefr_level', 'B1')}
- 需加強弱點：{', '.join(profile.get('weaknesses', []))}
- AI 系統難度調整指示：{profile.get('difficulty_note', '循序漸進')}

            請產出今日教材（請嚴格依據目前分數與難度指示調整用字深度與句型複雜度）：
1. 一篇約 230 字的英文短文（結合道地日常口語與多益商務情境）。
2. 5 個核心單字或片語（附繁體中文解釋與實用英文例句）。
3. 一段 8～10 輪的雙人 Podcast 對話劇本，由 Alex（男）與 Maya（女）輕鬆討論文章內容，加入自然口語連貫詞。speaker 只能填 "Alex" 或 "Maya"。
4. 2 題多益風格單選題（每題 4 個選項，附正確索引 0~3 與繁體中文解析）。
5. 2 個口說挑戰問題（第 1 題日常聊天，第 2 題多益職場情境）。
            """
            lesson_data = call_gemini_structured(
                prompt, DailyLesson, temperature=0.7
            )
            mp3_bytes = synthesize_podcast_mp3(lesson_data["podcast_script"])
            st.session_state["audio_bytes"] = mp3_bytes
            st.session_state["audio_date"] = today_str
            st.session_state["today_quiz_records"] = {}

            profile["latest_lesson"] = lesson_data
            profile["latest_lesson_date"] = today_str
            profile["latest_topic"] = lesson_data["topic_title"]
            profile["latest_questions"] = lesson_data["speaking_questions"]

            logs = [
                x
                for x in profile.get("lessons_log", [])
                if x["date"] != today_str
            ]
            logs.append({
                "date": today_str,
                "title": lesson_data["topic_title"],
                "category": lesson_data["category"],
                "toeic_score": profile["estimated_toeic"],
            })
            profile["lessons_log"] = logs

            vocab_bank = profile.get("vocab_bank", [])
            existing = {v["word"].lower() for v in vocab_bank}
            for v in lesson_data["vocabulary"]:
                if v["word"].lower() not in existing:
                    vocab_bank.append({
                        "word": v["word"],
                        "meaning_zh": v["meaning_zh"],
                        "date": today_str,
                    })
            profile["vocab_bank"] = vocab_bank
            save_profile(profile)
            st.rerun()

    if latest_lesson:
        st.caption(
            f"📅 課程日期：{profile.get('latest_lesson_date', today_str)} ｜ 🏷️ {latest_lesson['category']}"
        )
        st.subheader(latest_lesson["topic_title"])

        # Podcast 播放器
        st.markdown("#### 🎧 雙人 Podcast 訓練")
        if "audio_bytes" not in st.session_state:
            with st.spinner("🎙️ 正在載入今日 Podcast 語音..."):
                st.session_state["audio_bytes"] = synthesize_podcast_mp3(
                    latest_lesson["podcast_script"]
                )
        st.audio(st.session_state["audio_bytes"], format="audio/mp3")

        with st.expander("📜 點擊展開 100% 同步逐字稿"):
            for turn in latest_lesson["podcast_script"]:
                st.markdown(f"**{turn['speaker']}:** {turn['text']}")

        # 短文閱讀
        st.markdown("#### 📄 精選雙效短文")
        st.write(latest_lesson["article"])

        # 核心單字
        st.markdown("#### 📖 核心單字與片語")
        for v in latest_lesson["vocabulary"]:
            with st.container(border=True):
                st.markdown(f"**{v['word']}** — {v['meaning_zh']}")
                st.caption(f"例句：{v['example']}")

        # 隨堂測驗（自動記錄作答表現供 AI 評估）
        st.markdown("#### 📝 聽力與閱讀隨堂測驗")
        quiz_records = {}
        for idx, q in enumerate(latest_lesson["comprehension_quiz"], 1):
            st.markdown(f"**Q{idx}. {q['question']}**")
            choice = st.radio(
                f"選擇 Q{idx} 答案",
                q["options"],
                index=None,
                key=f"quiz_{profile.get('latest_lesson_date')}_{idx}",
                label_visibility="collapsed",
            )
            if choice is not None:
                correct_opt = q["options"][q["correct_index"]]
                is_correct = choice == correct_opt
                quiz_records[f"Q{idx}"] = {
                    "question": q["question"],
                    "is_correct": is_correct,
                }
                if is_correct:
                    st.success(f"✅ 答對了！{q['explanation_zh']}")
                else:
                    st.error(
                        f"❌ 正確答案是：{correct_opt}\n\n解析：{q['explanation_zh']}"
                    )
        st.session_state["today_quiz_records"] = quiz_records
    else:
        st.info(
            "👆 點擊上方按鈕，立即為你生成今天的第一堂隨身英文課！"
        )

# ==================== 分頁 2：口說實戰 ====================
with tab2:
    questions = profile.get("latest_questions", [])
    if not questions:
        st.info("請先到「📖 今日課程」生成課程，就會解鎖今天的口說挑戰題！")
    else:
        st.markdown("#### 🗣️ 通勤口說挑戰")
        st.caption(
            "方便開口時可直接按麥克風錄音；若在安靜車廂內，也可以直接用下方文字框輸入。送出後 AI 會自動評估你的真實程度並調整明日難度！"
        )

        audio_answers = []
        text_answers = []
        for idx, q in enumerate(questions, 1):
            with st.container(border=True):
                st.markdown(f"**【第 {idx} 題】{q}**")
                aud = st.audio_input(
                    f"🎙️ 錄製第 {idx} 題英文回答", key=f"aud_{idx}"
                )
                txt = st.text_input(
                    f"⌨️ 或輸入文字回答（第 {idx} 題）",
                    placeholder="I think...",
                    key=f"txt_{idx}",
                )
                audio_answers.append(aud)
                text_answers.append(txt)

        if st.button(
            "🚀 送出給 AI 教練客觀批改並自動調整難度",
            type="primary",
            use_container_width=True,
        ):
            has_any_input = any(a is not None for a in audio_answers) or any(
                t.strip() for t in text_answers
            )
            if not has_any_input:
                st.warning("請至少錄音或輸入一題的英文回答再送出喔！")
            else:
                with st.spinner("🧠 AI 教練正在客觀分析你的文法、用字深度與測驗表現..."):
                    quiz_records = st.session_state.get("today_quiz_records", {})
                    if quiz_records:
                        correct_cnt = sum(
                            1 for v in quiz_records.values() if v["is_correct"]
                        )
                        quiz_summary = f"今日閱讀測驗作答 {len(quiz_records)} 題，答對 {correct_cnt} 題（明細：{json.dumps(quiz_records, ensure_ascii=False)}）。"
                    else:
                        quiz_summary = "學生今日尚未作答閱讀測驗，請完全依據以下口說／文字回答表現進行客觀評估。"

                    contents = [
                        f"""
                        你是一位精準、客觀且善於因材施教的 AI 英文教練，目標是幫助學生「與外國人自然流暢聊天」並「考取多益 800 分以上」。
                        學生目前系統預估程度：多益 {profile['estimated_toeic']} 分（{profile.get('cefr_level', 'B1')}）。
                        【客觀測驗表現】{quiz_summary}

                        請不要依賴學生主觀感覺，而是**完全根據學生實際回答的英文品質（語音流暢度、發音清晰度、句型結構完整度、單字豐富度、文法正確率）以及測驗表現**，客觀判斷他目前的真實程度：
                        - 若回答非常順暢、句型多變、用字精準且測驗全對：給予大幅加分（score_delta: +15 到 +35），並在 next_difficulty_note 指示明日提高文章與單字難度。
                        - 若表現符合目前分數水準、有小文法錯誤但能溝通：給予穩定微調（score_delta: +5 到 +15），維持穩健難度。
                        - 若回答明顯吃力、大量中式英文、單字貧乏或測驗答錯：適度下修分數（score_delta: -20 到 -5），並在 next_difficulty_note 指示明日放慢難度、先鞏固基礎句型。
                        請在 ai_diagnostic_reason_zh 用繁體中文簡潔說明你做出此難度調整的客觀依據。
                        """
                    ]
                    for idx, q in enumerate(questions, 1):
                        contents.append(f"\n--- 第 {idx} 題題目：{q} ---")
                        if audio_answers[idx - 1] is not None:
                            aud_bytes = audio_answers[idx - 1].getvalue()
                            mime = audio_answers[idx - 1].type or "audio/wav"
                            contents.append("學生語音回答如下：")
                            contents.append(
                                types.Part.from_bytes(
                                    data=aud_bytes, mime_type=mime
                                )
                            )
                        if text_answers[idx - 1].strip():
                            contents.append(
                                f"學生文字回答：{text_answers[idx - 1].strip()}"
                            )

                    eval_data = call_gemini_structured(
                        contents, SpeakingEvaluation, temperature=0.5
                    )

                    old_score = profile["estimated_toeic"]
                    new_score = max(
                        300, min(990, old_score + eval_data["score_delta"])
                    )
                    new_cefr = score_to_cefr(new_score)
                    profile["estimated_toeic"] = new_score
                    profile["cefr_level"] = new_cefr
                    profile["weaknesses"] = eval_data["updated_weaknesses"]
                    profile["difficulty_note"] = eval_data[
                        "next_difficulty_note"
                    ]
                    profile["history"].append(
                        f"[{today_str}] AI 客觀評估：多益預估 {old_score} → {new_score} 分 ({new_cefr})"
                    )
                    save_profile(profile)

                    st.success(
                        f"🎉 AI 客觀評估完成！難度已自動校準：多益 {old_score} 分 → **{new_score} 分 ({new_cefr})**"
                    )
                    with st.container(border=True):
                        st.markdown("#### 🤖 AI 教練客觀程度診斷")
                        st.write(eval_data["ai_diagnostic_reason_zh"])
                        st.caption(f"🎯 明日課程難度方針：{eval_data['next_difficulty_note']}")

                    st.info(f"🌟 教練總評：{eval_data['overall_encouragement_zh']}")

                    for i, fb in enumerate(eval_data["feedbacks"], 1):
                        with st.container(border=True):
                            st.markdown(f"**📌 第 {i} 題講評**")
                            st.write(
                                f"👂 **聽取內容與修正：** {fb['transcript_or_summary_zh']}"
                            )
                            st.write(
                                f"🔍 **文法與自然度建議：** {fb['grammar_and_naturalness_zh']}"
                            )
                            st.success(
                                f"💬 **外國人日常聊法：** {fb['casual_native_rewrite']}"
                            )
                            st.info(
                                f"💼 **多益 850+ 高分示範：** {fb['toeic_850_rewrite']}"
                            )

# ==================== 分頁 3：學習儀表板 ====================
with tab3:
    score = profile.get("estimated_toeic", 550)
    col1, col2, col3 = st.columns(3)
    col1.metric("預估多益", f"{score} 分", profile.get("cefr_level", "B1"))
    col2.metric("完成堂數", f"{len(profile.get('lessons_log', []))} 堂")
    col3.metric("累積單字", f"{len(profile.get('vocab_bank', []))} 個")

    progress = min(1.0, max(0.05, score / 850.0))
    st.progress(progress, text=f"距離多益 800+ 目標進度 ({int(progress * 100)}%)")

    st.markdown("#### 🎯 AI 偵測重點加強方向")
    st.write("、".join(profile.get("weaknesses", [])))

    st.markdown("#### 🧭 目前課程難度方針")
    st.caption(profile.get("difficulty_note", "循序漸進提升口語自然度與多益商務理解"))

    st.markdown("#### 📖 累積單字複習庫")
    vocab_list = profile.get("vocab_bank", [])
    if vocab_list:
        for v in reversed(vocab_list):
            st.markdown(
                f"- **{v['word']}** ({v['meaning_zh']}) `{v.get('date', '')}`"
            )
    else:
        st.caption("尚無單字紀錄")

    st.markdown("#### 📈 成長與難度調整紀錄")
    for h in reversed(profile.get("history", [])):
        st.caption(f"• {h}")
