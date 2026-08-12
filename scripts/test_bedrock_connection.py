"""
Throwaway script - just proves AWS Bedrock connection works.
Not part of the real project, delete this after testing.
"""

import boto3

# Step A: List models you actually have access to (confirms Step A from
# earlier - the "Request access" click - actually worked)
bedrock = boto3.client("bedrock", region_name="us-east-1")
response = bedrock.list_foundation_models(byProvider="Anthropic")

print("=== Claude models you can access ===")
for model in response["modelSummaries"]:
    print(f"  {model['modelId']}")

# Step B: Actually call one of them and get a real answer back
runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# Pick the first available Claude model from the list above
#model_id = response["modelSummaries"][0]["modelId"]


model_id = "anthropic.claude-3-haiku-20240307-v1:0"
print(f"\n=== Calling model: {model_id} ===")

result = runtime.converse(
    modelId=model_id,
    messages=[
        {
            "role": "user",
            "content": [{"text": "Say hello in exactly 5 words."}],
        }
    ],
)

answer = result["output"]["message"]["content"][0]["text"]
print(f"\nAI replied: {answer}")