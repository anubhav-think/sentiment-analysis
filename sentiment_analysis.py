import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import pandas as pd


model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,return_all_scores = True)

#data =[text,text,..........]
def inference(data):
  test = []
  for i in data:
    text = i.replace('\n',' ')
    res = sentiment_task(text)
    test.append({'comment_text':text,res[0][0]['label']:res[0][0]['score'],res[0][1]['label']:res[0][1]['score'],res[0][2]['label']:res[0][2]['score']})

  df = pd.DataFrame(test)
  df.to_csv('output.csv')



def get_data(url):
  data =  requests.get(url)
  data = data.json()
  return data['data']

main = []
utc = "1680303707"
for i in range(100):
  url = f'https://api.pullpush.io/reddit/search/comment/?q=toyota&before={utc}'
  temp = get_data(url)
  for index,i in enumerate(temp):
    main.append(i['body'])
    if i==99:
      utc = i['created_utc']-1

inference(main)