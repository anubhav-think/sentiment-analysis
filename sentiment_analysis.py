import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import pandas as pd


model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,return_all_scores = True)

#data =[{'body':text},{'body':text},..........]
def inference(data):
  test = []
  for i in data:
    text = i['body'].replace('\n',' ')
    res = sentiment_task(text)
    test.append({'comment_text':text,res[0][0]['label']:res[0][0]['score'],res[0][1]['label']:res[0][1]['score'],res[0][2]['label']:res[0][2]['score']})

  df = pd.DataFrame(test)
  df.to_csv('output.csv')



url = 'https://api.pullpush.io/reddit/search/comment/?q=toyota&before=1680303707'

data =  requests.get(url)
data = data.json()

data = data['data']
inference(data)

