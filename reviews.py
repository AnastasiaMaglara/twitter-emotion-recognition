import pandas as pd
import json
import os
from emotion_predictor import EmotionPredictor

if not os.path.exists('data'):
    os.mkdir('data')

# Pandas presentation options
pd.options.display.max_colwidth = 100   # show whole tweet's content
pd.options.display.width = 200          # don't break columns
# pd.options.display.max_columns = 7      # maximal number of columns

# Predictor for Ekman's emotions in multiclass setting.
model = EmotionPredictor(classification='ekman', setting='mc', use_unison_model=True)


reviews = []

#forMoreJsonFiles
#data= []
#pathOfJsonFiles
#path_to_json = 'C:/Users/Αναστασία/Desktop/json/'
#for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
#  with open(path_to_json + file_name) as json_file:
#    data=json.load(json_file)
    # read list inside dict
    #_list = data['Reviews']
    # read listvalue and load dict
    #for v in _list:
    #   reviews.append(v['Content'])
    #predictions = model.predict_classes(reviews)
    #probabilities = model.predict_probabilities(reviews)
    #embeddings = model.embed(reviews)

#forOneJsonFile
with open('C:/Users/Αναστασία/Desktop/json/72572.json','r') as f:
  data = json.load(f)

# read list inside dict
_list = data['Reviews']
# read listvalue and load dict
for v in _list:
  reviews.append(v['Content'])

predictions = model.predict_classes(reviews)
print(predictions, '\n')

probabilities = model.predict_probabilities(reviews)
print(probabilities, '\n')

embeddings = model.embed(reviews)
print(embeddings, '\n')