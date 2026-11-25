import os
import time
import base64
import threading
from io import BytesIO

from flask import Flask, request, abort
from anthropic import Anthropic
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage
)

app = Flask(__name__)

# 讀取環境變數
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# 翻譯、糾錯、圖片辨識全部使用 Sonnet（品質較好）
SMART_MODEL = "claude-sonnet-4-6"

ERROR_REPLY = "抱歉，翻譯暫時出錯了，請稍後再試一次。"

# ---------- 長文設定 ----------
MAX_OUTPUT_TOKENS = 4000        # 長文章翻譯需要足夠的輸出長度，原本 1000 會寫到一半被截斷
LINE_CHUNK_CHARS = 4500         # LINE 單則文字上限 5000 字，保守切在 4500
LINE_MAX_MESSAGES = 5           # LINE 一次 reply 最多 5 則訊息

# ---------- 對話記憶設定 ----------
MAX_HISTORY_MESSAGES = 6        # 最多記住最近 3 輪（使用者 + bot 各算 1 則）
HISTORY_TTL_SECONDS = 30 * 60   # 超過 30 分鐘沒講話，就當成新對話
_history = {}                   # user_id -> {"messages": [...], "updated": 時間戳}
_history_lock = threading.Lock()

# ---------- 指令 ----------
CORRECTION_PREFIXES = ("改：", "改:", "改 ")
RESET_COMMANDS = ("清除", "重置", "/reset")

TRANSLATE_SYSTEM = (
    "你是一個專業的韓語翻譯與教學助手。請先判斷使用者最新一則文字的語言，再依下列規則回覆。\n\n"
    "【如果是中文句子】請翻譯成韓文，並分別列出：\n"
    "1. 敬語（합니다體／格式體，正式場合使用）\n"
    "2. 書面語（文章、書信等書面表達使用的形式）\n"
    "3. 半語（朋友、晚輩間使用的非正式口語形式）\n"
    "每一種韓文後面都用一兩句話說明適用場合。\n\n"
    "【如果是韓文句子】請翻譯成繁體中文，列出句子中的重點單字（韓文與中文意思），"
    "並簡短說明用到的文法。\n\n"
    "【如果是很長的文章（多個段落）】請依原文段落順序完整翻譯每一段，不可省略或只翻前面；"
    "重點單字只挑最重要的 8 到 10 個，文法說明精簡成 3 點以內，確保全文翻譯一定寫完。\n\n"
    "【如果是接續上一則的追問】例如「改成半語」「再正式一點」「這個單字是什麼意思」，"
    "請根據前面的對話內容直接回答追問，不需要重新列出三種形式。\n\n"
    "格式要求：使用繁體中文與純文字段落，切勿使用表格，也不要使用 Markdown 符號（例如 ** 或 #）。"
)

CORRECTION_SYSTEM = (
    "你是一位嚴謹又親切的韓語寫作老師，學生是以中文為母語、正在準備 TOPIK 的學習者。"
    "學生會給你一段自己寫的韓文，請依下列格式回覆：\n"
    "1. 修正後的句子（若原句已經正確，就明確說「這句是正確的」，並可附上一個更自然的說法）\n"
    "2. 錯誤說明：逐一指出錯在哪裡，每一項包含「原本的寫法」「問題」「對應的文法或用法規則」，"
    "並盡量用一個簡短例句幫助理解\n"
    "3. 一句話的總評與小建議\n\n"
    "如果學生傳來的內容不是韓文，請請他在「改：」後面貼上自己寫的韓文。\n"
    "格式要求：使用繁體中文與純文字段落，切勿使用表格，也不要使用 Markdown 符號（例如 ** 或 #）。"
)

IMAGE_PROMPT = (
    "你是一個專業的韓語翻譯與教學助手，使用者是以中文為母語、正在學韓文的人。"
    "請先判斷這張圖片屬於下列哪一種情況，再依規則回覆：\n\n"
    "【情況一：圖片中有韓文文字】（招牌、菜單、標籤、對話截圖等）\n"
    "1. 先列出你辨識到的原始韓文文字\n"
    "2. 提供繁體中文翻譯\n"
    "3. 列出重點單字（韓文與中文意思），並附上常用口語形式（-요 形）與簡短說明\n\n"
    "【情況二：圖片中有中文或其他語言的文字】（中文菜單、告示、對話截圖等）\n"
    "1. 先列出辨識到的原文\n"
    "2. 翻譯成自然的韓文，並在每句後面附上繁體中文意思\n\n"
    "【情況三：圖片中沒有文字，是照片】（食物、風景、物品、人物活動等）\n"
    "不要說「無法翻譯」，而是用韓文描述這張圖片，請依序提供：\n"
    "1. 用 1 到 3 句簡單的韓文描述圖片內容（適合 TOPIK 初中級學習者），每句後面附上繁體中文意思\n"
    "2. 圖片中主要物品的韓文單字（例如食物的韓文名稱），格式為「韓文：中文」\n\n"
    "格式要求：使用繁體中文與純文字段落，切勿使用表格，也不要使用 Markdown 符號（例如 ** 或 #）。"
    "韓文內容只寫韓文字，不要加羅馬拼音。"
)


@app.route("/", methods=['GET'])
def index():
    return "LINE Korean Translator Bot is Running!"


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


def ask_claude(model, messages, system=None):
    """呼叫 Claude。成功回傳文字，失敗回傳 None（由呼叫端決定怎麼回覆）。"""
    try:
        kwargs = {"model": model, "max_tokens": MAX_OUTPUT_TOKENS, "messages": messages}
        if system:
            kwargs["system"] = system
        response = claude_client.messages.create(**kwargs)
        text = response.content[0].text
        if response.stop_reason == "max_tokens":
            text += "\n\n（內容太長，後半段被截斷了。可以把文章分成幾段再傳。）"
        return text
    except Exception as e:
        app.logger.error("Claude API error: %s", e)
        return None


def get_history(user_id):
    """取得這位使用者最近的對話；太久沒講話就回傳空的。"""
    with _history_lock:
        entry = _history.get(user_id)
        if not entry or time.time() - entry["updated"] > HISTORY_TTL_SECONDS:
            return []
        return list(entry["messages"])


def save_history(user_id, history, user_text, reply_text):
    """把這一輪對話存起來，並順手清掉過期的紀錄。"""
    messages = (history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply_text},
    ])[-MAX_HISTORY_MESSAGES:]
    now = time.time()
    with _history_lock:
        _history[user_id] = {"messages": messages, "updated": now}
        expired = [u for u, e in _history.items()
                   if now - e["updated"] > HISTORY_TTL_SECONDS]
        for u in expired:
            del _history[u]


def clear_history(user_id):
    with _history_lock:
        _history.pop(user_id, None)


def parse_correction(text):
    """若訊息以「改：」開頭，回傳後面的韓文內容；否則回傳 None。"""
    for prefix in CORRECTION_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return None


def detect_media_type(data):
    """依檔案開頭判斷圖片格式，預設為 jpeg。"""
    if data.startswith(b'\x89PNG'):
        return "image/png"
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image/webp"
    if data[:3] == b'GIF':
        return "image/gif"
    return "image/jpeg"


def split_text(text, limit=LINE_CHUNK_CHARS):
    """把長文字依段落切成多則，每則不超過 limit 字。"""
    chunks, current = [], ""
    for para in text.split("\n"):
        # 單一段落本身就超過上限時，硬切
        while len(para) > limit:
            if current:
                chunks.append(current.rstrip("\n"))
                current = ""
            chunks.append(para[:limit])
            para = para[limit:]
        if len(current) + len(para) + 1 > limit:
            chunks.append(current.rstrip("\n"))
            current = ""
        current += para + "\n"
    if current.strip():
        chunks.append(current.rstrip("\n"))
    return chunks or [""]


def reply(event, text):
    chunks = split_text(text)
    if len(chunks) > LINE_MAX_MESSAGES:
        chunks = chunks[:LINE_MAX_MESSAGES]
        chunks[-1] = chunks[-1][:LINE_CHUNK_CHARS - 30] + "\n\n（內容過長，已截斷）"
    messages = [TextSendMessage(text=c) for c in chunks]
    try:
        line_bot_api.reply_message(event.reply_token, messages)
    except Exception as e:
        # reply token 逾時（長文生成較久）時，改用 push 補送
        app.logger.error("reply_message failed, fallback to push: %s", e)
        try:
            line_bot_api.push_message(event.source.user_id, messages)
        except Exception as e2:
            app.logger.error("push_message failed: %s", e2)


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_text = event.message.text.strip()
    user_id = event.source.user_id

    # 指令：清除對話記憶
    if user_text in RESET_COMMANDS:
        clear_history(user_id)
        reply(event, "已清除對話記憶，可以開始新的話題。")
        return

    history = get_history(user_id)

    # 糾錯模式：「改：我是學生이에요」
    correction = parse_correction(user_text)
    if correction is not None:
        if not correction:
            reply(event, "請在「改：」後面貼上你寫的韓文，例如：\n改：저는 어제 학교에 갔어요")
            return
        stored_text = "（請幫我校正這句韓文）" + correction
        result = ask_claude(
            SMART_MODEL,
            history + [{"role": "user", "content": stored_text}],
            system=CORRECTION_SYSTEM,
        )
    else:
        stored_text = user_text
        result = ask_claude(
            SMART_MODEL,
            history + [{"role": "user", "content": stored_text}],
            system=TRANSLATE_SYSTEM,
        )

    if result is None:
        reply(event, ERROR_REPLY)
        return

    save_history(user_id, history, stored_text, result)
    reply(event, result)


@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    user_id = event.source.user_id
    history = get_history(user_id)
    result = None
    try:
        # 從 LINE 下載使用者傳來的圖片內容
        message_content = line_bot_api.get_message_content(event.message.id)
        image_bytes = BytesIO()
        for chunk in message_content.iter_content():
            image_bytes.write(chunk)
        raw = image_bytes.getvalue()

        image_b64 = base64.b64encode(raw).decode('utf-8')
        media_type = detect_media_type(raw)

        result = ask_claude(
            SMART_MODEL,
            [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": IMAGE_PROMPT},
                ],
            }],
        )
    except Exception as e:
        app.logger.error("Image handling error: %s", e)
        result = None

    if result is None:
        reply(event, ERROR_REPLY)
        return

    # 把這次圖片翻譯存進對話記憶，之後可以接著追問（例如「這個單字是什麼意思」）
    save_history(user_id, history, "（使用者傳了一張圖片請你翻譯）", result)
    reply(event, result)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
