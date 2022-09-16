import pandas as pd 
f = pd.read_csv('tam_test.tsv', '\t')
f['category'] = ['Positive']*len(f)
#f.pop('i')
#f= f.reset_index(drop=True)

f.to_csv('NEW_TAM_TEST.tsv', sep='\t',columns=['text', 'category'], index=False)