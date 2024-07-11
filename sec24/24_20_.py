import pandas as pd
import nlplot
import re
from janome.tokenizer import Tokenizer

def wakachigaki(text):
    token = Tokenizer().tokenize(text)
    words = []
    for line in token:
        tkn = re.split('\t|,', str(line))
        # tkn[0]が存在し、tkn[1]が名詞でtkn[2]が一般または固有名詞であるか
        if tkn[0] and tkn[1] in ['名詞'] and tkn[2] in ['一般', '固有名詞'] :
            words.append(tkn[0])
        #words.append(tkn[0])
    return ' ' . join(words)

text = open("soseki.txt", encoding="utf8").read()
text = wakachigaki(text)
print(text)
df = pd.read_table('soseki_wakachi.txt', names=['tokens'])

# target_col as a list type or a string separated by a space.
npt = nlplot.NLPlot(df, target_col='tokens')
stopwords = npt.get_stopword(top_n=15, min_freq=2)
npt.build_graph(stopwords=stopwords, min_edge_frequency=5)
npt.co_network(title='Co-occurrence network', width=600, height=500, save=True)
