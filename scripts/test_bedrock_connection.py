"""
Quick throwaway check - is AWS Bedrock working right now?
Tests both an embedding call and a text-generation call.
"""

import boto3

REGION = "us-east-1"  # change if your Bedrock access is in a different region

print("=== 1. Testing EMBEDDING model (needed for Knowledge Base) ===")
runtime = boto3.client("bedrock-runtime", region_name=REGION)

import json

try:
    # Try Cohere first (you have this), fall back to Titan
    response = runtime.invoke_model(
        modelId="cohere.embed-english-v3",
        body=json.dumps({"texts": ["test"], "input_type": "search_document"}),
    )
    print("✅ Cohere embedding works")
except Exception as e:
    print(f"❌ Cohere embedding failed: {e}")

print("\n=== 2. Testing TEXT model (Claude) ===")
try:
    result = runtime.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[{"role": "user", "content": [{"text": "Say hello in 3 words."}]}],
    )
    answer = result["output"]["message"]["content"][0]["text"]
    print(f"✅ Claude works — replied: {answer}")
except Exception as e:
    print(f"❌ Claude failed: {e}")