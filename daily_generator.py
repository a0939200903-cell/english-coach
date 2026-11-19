import datetime
import json
import os
import time
from google import genai
from google.genai import types
from pydantic import BaseModel

PROFILE_FILE = "student_profile.json"
TEXT_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
]


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


def load_profile() -> dict:
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "estimated_toeic": 550,
        "cefr_level": "B1",
        "strengths": ["Business Vocabulary"],
        "weaknesses": ["Daily Conversational English", "Advanced Grammar"],
        "difficulty_note": "循序漸進提升口語自然度與多益商務理解",
        "lessons_log": [],
        "vocab_bank": [],
        "last_assessment_lesson_count": 0,
        "history": ["建立雲端學習檔案（起始預估多益 550 分）"],
    }


def generate_daily_lesson(profile: dict) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("找不到 GEMINI_API_KEY 環境變數")

    client = genai.Client(api_key=api_key)
    prompt = f"""
    你是一位能動態調整教學難度的專屬英文家教。
    【學生檔案】
    - 目前預估多益分數：{profile.get('estimated_toeic', 550)} 分（目標 800 分以上，並能流暢與外國人日常聊天）
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

    for _ in range(3):
        for model_name in TEXT_MODELS:
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=DailyLesson,
                        temperature=0.7,
                    ),
                )
                return json.loads(resp.text)
            except Exception as e:
                print(f"模型 {model_name} 暫時無法回應，重試中... ({e})")
                time.sleep(2)

    raise RuntimeError("所有 Gemini 模型皆暫時忙碌，本次排程失敗。")


def main():
    tz_tw = datetime.timezone(datetime.timedelta(hours=8))
    today_str = datetime.datetime.now(tz_tw).strftime("%Y-%m-%d")

    profile = load_profile()
    lesson_data = generate_daily_lesson(profile)

    profile["latest_lesson"] = lesson_data
    profile["latest_lesson_date"] = today_str
    profile["latest_topic"] = lesson_data["topic_title"]
    profile["latest_questions"] = lesson_data["speaking_questions"]

    logs = [x for x in profile.get("lessons_log", []) if x.get("date") != today_str]
    logs.append({
        "date": today_str,
        "title": lesson_data["topic_title"],
        "category": lesson_data["category"],
        "toeic_score": profile.get("estimated_toeic", 550),
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

    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"✅ 已成功生成 {today_str} 專屬課程：{lesson_data['topic_title']}")


if __name__ == "__main__":
    main()
