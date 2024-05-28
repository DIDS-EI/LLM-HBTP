
from openai import OpenAI



class LLMGPT3():
    def __init__(self):
        self.client = OpenAI(
            # base_url="YourURL", api_key="sk-YourKey"
            base_url="https://gtapi.xiaoerchaoren.com:8932/v1",
            api_key="sk-OO5BXh9SUMrnWR6q6fC035142aC94352A59f78E8655fE62b"
        )
    def request(self,message): # question
        completion = self.client.chat.completions.create(
          model="gpt-3.5-turbo",
          # message
            messages=message
        )

        return completion.choices[0].message.content

    def embedding(self,question):
        embeddings = self.client.embeddings.create(
          model="text-embedding-3-small",
          input=question
        )

        return embeddings
    def list_models(self):
        response = self.client.models.list()
        return response.data
    def list_embedding_models(self):
        models = self.list_models()
        embedding_models = [model.id for model in models if "embedding" in model.id]
        return embedding_models


if __name__ == '__main__':
    llm = LLMGPT3()

    answer = llm.embedding(question="who are you,gpt?")
    print(answer)

