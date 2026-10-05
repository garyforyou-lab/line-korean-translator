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

TEXT_PROMPT = (
    "你是一個專業的韓語翻譯與教學助手。使用者會傳送中文（也可能是韓文）文字給你，"
    "請將其翻譯成韓文，並分別列出以下三種形式：\n"
    "1. 敬語（합니다體／格式體，正式場合使用）\n"
    "2. 書面語（文章、書信等書面表達使用的形式）\n"
    "3. 半語（朋友、晚輩間使用的非正式口語形式）\n"
    "每種形式請附上簡短說明，說明請使用純文字段落格式，切勿使用表格。"
    "文字如下：\n{user_text}"
)

IMAGE_PROMPT = (
    "你是一個專業的韓語翻譯與教學助手。請閱讀這張圖片中出現的所有韓文文字"
    "（例如招牌、菜單、標籤、對話截圖等），並：\n"
    "1. 先列出你辨識到的原始韓文文字\n"
    "2. 提供繁體中文翻譯\n"
    "3. 附上常用口語形式（-요 形）與簡短說明\n"
    "說明請使用純文字段落格式，切勿使用表格。"
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


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_text = event.message.text.strip()

    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": TEXT_PROMPT.format(user_text=user_text)}]
    )

    reply_text = response.content[0].text
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))


@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    # 從 LINE 下載使用者傳來的圖片內容
    message_content = line_bot_api.get_message_content(event.message.id)
    image_bytes = BytesIO()
    for chunk in message_content.iter_content():
        image_bytes.write(chunk)
    image_bytes.seek(0)

    # 轉成 base64，供 Claude vision 使用
    image_b64 = base64.b64encode(image_bytes.read()).decode('utf-8')

    # LINE 傳來的圖片內容一般為 jpeg
    media_type = "image/jpeg"

    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[
            {
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
            }
        ],
    )

    reply_text = response.content[0].text
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
