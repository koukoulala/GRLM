import msal
import requests
from azure.identity import AzureCliCredential, DefaultAzureCredential

papyrus_endpoint = "https://westus2batch.papyrus.binginternal.com/chat/completions"

verify_scope = "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default"

#cur_credential = AzureCliCredential()
cur_credential = DefaultAzureCredential()

access_token = cur_credential.get_token(verify_scope).token

# Add your papyrus quota id here or use "papyruscustomer" after you have joined the papyrus customer role for local testing
papyrus_quota_id = ""

endpoint = "https://WestUS2Batch.papyrus.binginternal.com/evalandbatchmodels"
result = requests.get(endpoint)
print(result.json())

# Add the papyrus model name you want to access
#papyrus_model_name = "gpt-54-2026-03-05-Eval"
#papyrus_model_name = "gpt-52-2025-12-11-Eval"
papyrus_model_name = "gpt-51-chat-txt-shortco-2025-11-13-Batch"

headers = {
    "Authorization": "Bearer " + access_token,
    "Content-Type": "application/json",
    "papyrus-model-name": papyrus_model_name,
    "papyrus-quota-id": papyrus_quota_id,
    "papyrus-timeout-ms": "100000",
    }
 
json_dict = {"messages":[{
                "role": "system",
                "content": "You are a jokester. You always respond sarcastically before answering a question with factual answers."},
                {
                "role": "user",
                "content": "How to cook fish?"}],
            "max_completion_tokens": 500}
 
response = requests.post(papyrus_endpoint, headers=headers, json=json_dict)
print(response.status_code)
print(response.headers)
print(response.text)

# Extract the answer content
import json
try:
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    print("\n" + "=" * 70)
    print("Answer:")
    print("=" * 70)
    print(content)
except (json.JSONDecodeError, KeyError, IndexError) as e:
    print(f"\nFailed to extract content: {e}")