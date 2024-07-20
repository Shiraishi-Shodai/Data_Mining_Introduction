import nlplot
import pandas as pd
import plotly
from plotly.subplots import make_subplots
from plotly.offline import iplot
import matplotlib.pyplot as plt

df = pd.read_csv('sample_twitter.csv')
npt = nlplot.NLPlot(df, target_col='hashtags')
# 今回は上位2単語をストップワードに指定(ここではデータサイエンティストやkaggleをストップワードにしている)
stopwords = npt.get_stopword(top_n=2, min_freq=0)
npt.build_graph(stopwords=stopwords, min_edge_frequency=25)
  
fig_co_network = npt.co_network(
    title='Co-occurrence network',
    sizing=100,
    node_size='adjacency_frequency',
    color_palette='hls',
    width=1100,
    height=700,
    save=False
)
iplot(fig_co_network)
