import nlplot
import pandas as pd
import plotly
from plotly.subplots import make_subplots
from plotly.offline import iplot
import matplotlib.pyplot as plt

df = pd.read_csv('sample_twitter.csv')
npt = nlplot.NLPlot(df, target_col='hashtags')
# 今回は上位2単語をストップワードに指定
stopwords = npt.get_stopword(top_n=2, min_freq=0)

fig_wc = npt.wordcloud(
    width=1000,
    height=600,
    max_words=100,
    max_font_size=100,
    colormap='tab20_r',
    stopwords=stopwords,
    mask_file=None,
    save=False,
)
plt.figure(figsize=(15, 25))
plt.imshow(fig_wc, interpolation="bilinear")
plt.axis("off")
plt.show()

