from openai import OpenAI

client = OpenAI(api_key="")
models = client.models.list()
print(len(models.data))


