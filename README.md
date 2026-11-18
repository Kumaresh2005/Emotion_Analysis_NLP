<div align="center">
# 🧠 Emotion Detector
 
### Type any sentence and let NLP tell you how it feels.
 
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Open_App-8E7CFF?style=for-the-badge)](https://emotionanalysiss.streamlit.app/)
 
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-2E7D32?style=flat-square)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-~86%25-FFD166?style=flat-square)
 
**[👉 Try the live app here](https://emotionanalysiss.streamlit.app/)**
 
</div>
---
 
## ✨ About the Project
 
**Emotion Detector** is an NLP web app that reads a sentence and predicts the emotion behind it, with a confidence score for every emotion.
 
I built the machine learning model while learning NLP in a Jupyter Notebook, then turned it into a full interactive web app using **Streamlit**.
 
### 😄 Emotions it can detect
 
| Emotion | Emoji | Example |
|---|---|---|
| Joy | 😄 | *"i am so happy today because i passed my exam"* |
| Love | 😍 | *"i feel so much love and care for my family"* |
| Surprise | 😲 | *"i was shocked and amazed when i saw the surprise party"* |
| Sadness | 😢 | *"i feel so lonely and hopeless after everything"* |
| Anger | 😡 | *"i am furious that they ignored my complaint again"* |
| Fear | 😨 | *"i am terrified about the results coming out tomorrow"* |
 
---
 
## 🎯 Features
 
- ✍️ **Analyze text** – type a sentence and get the predicted emotion with a big emoji result card
- 📊 **Probability bars** – see how likely each of the 6 emotions is
- 🧹 **Cleaned text view** – see exactly what the model reads after preprocessing
- 📁 **Batch analysis** – paste many sentences or upload a CSV, then download the results
- 📈 **Dataset insights** – charts and samples from the training data
- ⚡ **One-click examples** – try the app instantly from the sidebar
- 🎨 **Modern dark UI** – gradient background, glass-style cards and smooth animations
---
 
## ⚙️ How It Works
 
```
Raw text
   │
   ▼
1. Convert to lowercase
2. Remove punctuation
3. Remove numbers
4. Remove emojis / non-ASCII characters
5. Remove stop words (NLTK)
   │
   ▼
TF-IDF Vectorizer  ──►  Logistic Regression  ──►  Emotion + Confidence
```
 
I tried three approaches in the notebook and chose the best one:
 
| Vectorizer | Model | Result |
|---|---|---|
| Bag of Words | Naive Bayes | Lower |
| TF-IDF | Naive Bayes | Lower |
| **TF-IDF** | **Logistic Regression** | **Best (~86%)** ✅ |
 
---
 
## 🛠️ Tech Stack
 
| Area | Tools |
|---|---|
| Language | Python |
| Web app | Streamlit, HTML/CSS |
| Machine learning | scikit-learn (TF-IDF, Logistic Regression) |
| NLP | NLTK (stop words) |
| Data | Pandas |
 
---
 
## 📂 Project Structure
 
```
Emotion_Analysis_NLP/
├── .streamlit/
│   └── config.toml        # Dark theme settings
├── app.py                 # Streamlit app (UI + model)
├── train.txt              # Dataset (16,000 labeled sentences)
├── requirements.txt       # Python libraries
└── README.md
```
 
---
 
## 🚀 Run It Locally
 
**1. Clone the repository**
```bash
git clone https://github.com/Kumaresh2005/Emotion_Analysis_NLP.git
cd Emotion_Analysis_NLP
```
 
**2. Install the libraries**
```bash
pip install -r requirements.txt
```
 
**3. Start the app**
```bash
streamlit run app.py
```
 
The app opens at `http://localhost:8501`. The first run trains the model, which takes a few seconds. After that it is cached.
 
---
 
## 📊 Dataset
 
- **16,000** English sentences, each labeled with one of 6 emotions
- Format: `sentence;emotion` (separated by a semicolon)
- Class counts: joy 5,362 · sadness 4,666 · anger 2,159 · fear 1,937 · love 1,304 · surprise 572
---
 
## ⚠️ Limitations
 
- The dataset is **imbalanced**, so the model is weaker on *surprise* and *love* than on *joy* or *sadness*.
- Stop-word removal also removes words like *"not"*, so negation (e.g. *"i am not happy"*) can be misread.
- It is a classic ML model, not a deep learning model, so it can miss sarcasm and complex context.
## 🔮 Future Improvements
 
- [ ] Keep negation words to handle sentences like "not happy"
- [ ] Balance the classes to improve *surprise* and *love*
- [ ] Try a transformer model (BERT / DistilBERT) for higher accuracy
- [ ] Add a confusion matrix and F1 score to the insights tab
---
 
## 👨‍💻 Author
 
**Kumaresh Biswas**
B.Tech Electronics & Communication Engineering student · Data Science and AI learner
 
[![GitHub](https://img.shields.io/badge/GitHub-Kumaresh2005-181717?style=flat-square&logo=github)](https://github.com/Kumaresh2005)
 
---
 
<div align="center">
⭐ If you liked this project, please give it a star! ⭐
 
Built with ❤️ using Streamlit, scikit-learn and NLTK
 
</div>
 
