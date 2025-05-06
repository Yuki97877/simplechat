import nest_asyncio
nest_asyncio.apply()
from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import boto3
import re
from botocore.exceptions import ClientError
import uvicorn
from pyngrok import ngrok

# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値

# FastAPIの初期化
app = FastAPI()

# モデルID
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

# クライアントの初期化処理
bedrock_client = None

# 推論のためのAPIエンドポイント
class MessageRequest(BaseModel):
    message: str
    conversationHistory: list

@app.get("/")
async def home():
    return {"message": "Welcome to the FastAPI app!"}

@app.post("/predict/")
async def predict(request: MessageRequest):
    try:
        region = "us-east-1"
        global bedrock_client
        if bedrock_client is None:
            bedrock_client = boto3.client('bedrock-runtime', region_name=region)

        messages = request.conversationHistory.copy()
        messages.append({"role": "user", "content": request.message})

        bedrock_messages = [{"role": msg["role"], "content": [{"text": msg["content"]}]} for msg in messages]

        request_payload = {
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": 512,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9
            }
        }

        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_payload),
            contentType="application/json"
        )

        response_body = json.loads(response['body'].read())

        if not response_body.get('output') or not response_body['output'].get('message') or not response_body['output']['message'].get('content'):
            raise Exception("No response content from the model")

        assistant_response = response_body['output']['message']['content'][0]['text']

        messages.append({"role": "assistant", "content": assistant_response})

        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "success": False,
                "error": str(e)
            })
        }

# Google Colab用のFastAPI起動設定
if __name__ == "__main__":
    port = 8000
    public_url = ngrok.connect(port)
    print(f"FastAPI is now live at {public_url}")  # URLを表示
    uvicorn.run(app, host="0.0.0.0", port=port)
