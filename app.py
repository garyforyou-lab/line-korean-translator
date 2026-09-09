import os
from flask import Flask, request, abort
from anthropic import Anthropic
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)

# 讀取環境變數
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)

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
def handle_message(event):
    user_text = event.message.text.strip()
    
    # 呼叫 Claude 進行韓文翻譯與解析
    prompt = f"你是一個專業的韓語翻譯與教學助手。請將使用者發送的文字翻譯成韓文，並附上常用口語形式（-요 形）與簡短說明（說明請使用純文字段落格式，切勿使用表格）。文字如下：\n{user_text}"
    
    response = claude_client.messages.create(
        model="claude-sonnet-4-6t",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    reply_text = response.content[0].text
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
