# Sarcasm Detection with RNN & LSTM

Deep learning text classification of news headlines as **sarcastic** or **not sarcastic**. Compares three architectures built with TensorFlow/Keras, and includes a Gradio app for real-time predictions.

## Results

Evaluated on a held-out 20% test set (5,724 headlines).

| Model | Test Accuracy | Test Loss |
|---|---|---|
| SimpleRNN | 79.79% | 0.4386 |
| LSTM | 78.97% | 0.4483 |
| **LSTM + pretrained word2vec** | **82.16%** | **0.3993** |

Pretrained word2vec embeddings gave the best result (F1 ≈ 0.82 on both classes).

## Dataset

- `sarcastic_headlines.csv`: 28,619 headlines with a binary `is_sarcastic` label (14,985 not sarcastic, 13,634 sarcastic)
- The dataset is not included in this repo; add it to your working directory and update the path in the notebook.

## Pipeline

1. **Cleaning**: expand contractions, lowercase, strip URLs/mentions/hashtags/numbers/punctuation, remove stopwords, lemmatize (NLTK)
2. **Exploration**: word clouds per class and top-20 word frequencies
3. **Preprocessing**: stratified 80/20 train/test split, Keras tokenizer (20,000-word vocab, OOV token), post-padding to length 11 (95th percentile)
4. **Models**
   - SimpleRNN with a trainable 64-d embedding
   - LSTM (128 units) with dropout and a 64-d embedding
   - LSTM (128 units) with dropout, initialised from 300-d Google News word2vec (16,268 of 20,000 words covered)
5. **Training**: Adam, binary cross-entropy, batch size 64, up to 15 epochs, early stopping on validation loss
6. **Evaluation**: accuracy, confusion matrices, precision/recall/F1
7. **Demo**: Gradio interface that returns each model's prediction and confidence for any input headline

## Tech Stack

Python, TensorFlow/Keras, NLTK, gensim, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, WordCloud, Gradio

## Getting Started

```bash
git clone https://github.com/aayush505/<repo-name>.git
cd <repo-name>
pip install numpy pandas matplotlib seaborn nltk gensim wordcloud contractions scikit-learn tensorflow gradio
jupyter notebook sarcasm_detection.ipynb
```

The notebook was developed in Google Colab. The word2vec vectors (`word2vec-google-news-300`, ~1.6 GB) download automatically through `gensim` on first run.

## Author

**Aayush Silwal**: [GitHub](https://github.com/aayush505)
