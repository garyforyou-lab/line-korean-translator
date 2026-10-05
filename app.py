import os
import base64
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

# 文字翻譯用較便宜的 Haiku，圖片辨識用 Sonnet
TEXT_MODEL = "claude-haiku-4-5-20251001"
IMAGE_MODEL = "claude-sonnet-4-6"

ERROR_REPLY = "抱歉，翻譯暫時出錯了，請稍後再試一次。"

TEXT_PROMPT = (
    "你是一個專業的韓語翻譯與教學助手。請先判斷使用者文字的語言，再依下列規則回覆。\n\n"
    "【如果是中文】請翻譯成韓文，並分別列出：\n"
    "1. 敬語（합니다體／格式體，正式場合使用）\n"
    "2. 書面語（文章、書信等書面表達使用的形式）\n"
    "3. 半語（朋友、晚輩間使用的非正式口語形式）\n"
    "每一種韓文後面都用一兩句話說明適用場合。\n\n"
    "【如果是韓文】請翻譯成繁體中文，列出句子中的重點單字（韓文與中文意思），"
    "並簡短說明用到的文法。\n\n"
    "格式要求：使用純文字段落，切勿使用表格，也不要使用 Markdown 符號（例如 ** 或 #）。\n\n"
    "使用者的文字如下：\n{user_text}"
)

IMAGE_PROMPT = (
    "你是一個專業的韓語翻譯與教學助手。請閱讀這張圖片中出現的所有韓文文字"
    "（例如招牌、菜單、標籤、對話截圖等），並：\n"
    "1. 先列出你辨識到的原始韓文文字\n"
    "2. 提供繁體中文翻譯\n"
    "3. 附上常用口語形式（-요 形）與簡短說明\n"
    "格式要求：使用純文字段落，切勿使用表格，也不要使用 Markdown 符號（例如 ** 或 #）。"
    "如果圖片中沒有韓文文字，請告知使用者。"
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


def ask_claude(model, content):
    """呼叫 Claude，出錯時回傳提示文字，避免 bot 完全沒反應。"""
    try:
        response = claude_client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text
    except Exception as e:
        app.logger.error("Claude API error: %s", e)
        return ERROR_REPLY


def detect_media_type(data):
    """依檔案開頭判斷圖片格式，預設為 jpeg。"""
    if data.startswith(b'\x89PNG'):
        return "image/png"
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image/webp"
    if data[:3] == b'GIF':
        return "image/gif"
    return "image/jpeg"


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_text = event.message.text.strip()

    reply_text = ask_claude(
        TEXT_MODEL,
        TEXT_PROMPT.format(user_text=user_text),
    )

    # LINE 單則文字上限 5000 字
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text=reply_text[:4900])
    )


@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    try:
        # 從 LINE 下載使用者傳來的圖片內容
        message_content = line_bot_api.get_message_content(event.message.id)
        image_bytes = BytesIO()
        for chunk in message_content.iter_content():
            image_bytes.write(chunk)
        raw = image_bytes.getvalue()

        image_b64 = base64.b64encode(raw).decode('utf-8')
        media_type = detect_media_type(raw)

        reply_text = ask_claude(
            IMAGE_MODEL,
            [
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
        )
    except Exception as e:
        app.logger.error("Image handling error: %s", e)
        reply_text = ERROR_REPLY

    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text=reply_text[:4900])
    )


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
