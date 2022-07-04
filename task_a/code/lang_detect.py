from langdetect import detect_langs
from langdetect import detect
import pandas as pd

tamil = pd.read_csv('../data/tam_sentiment_train.tsv','\t')

langs = []

f = open('tamil_langs.txt', 'w')
for i in tamil['text']:
    if len(i) > 0:
        langs.append(detect(i))
        f.write(str(langs[-1]) + '\n')
    print(langs)
f.close()