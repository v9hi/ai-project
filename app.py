import gradio as gr
import pandas as pd
from transformers import pipeline
import nltk
import re
import string
from nltk.corpus import stopwords

pipe = pipeline("text-classification", model="Am09/distilroberta-base-fake_news_detector-Am09")
nltk.download('stopwords')
stopwords_english = stopwords.words('english')

def remove_punct(text):
    return ("".join([ch for ch in text if ch not in string.punctuation]))

def tokenize(text):
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]

def remove_small_words(text):
    return [x for x in text if len(x) > 3 ]

def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]

def return_sentences(tokens):
    return " ".join([word for word in tokens])

def preprocessing_pipeline(text):
    text = remove_punct(text)
    text = tokenize(text)
    text = remove_small_words(text)
    text = return_sentences(text)
    return text

def submit(title, text, file):

   if ((title == "" and text == "") and (file is None)):
        return "Error: Please enter either title and text or upload a CSV file."

   if file is not None:
        if not file.name.endswith('.csv'):
            return "Error: Uploaded file is not in CSV format."
        
        try:
            df = pd.read_csv(file)
            df['id'] = range(1, len(df) + 1)
            df["text"] = df["title"] + df["text"]
            df.drop("title", axis=1, inplace=True)
            predictions = []
            for index, row in df.iterrows():
                input_text= row["text"]
                input_text=preprocessing_pipeline(input_text)
                input_text = input_text[:512] if len(input_text) > 512 else input_text
                result = pipe(input_text)
                label_id = result[0]["label"]
                if label_id == "LABEL_0":
                    prediction = "Fake"
                else:
                    prediction = "True"
                predictions.append((row['id'], prediction))
            output = "\n".join([f"{id}: {prediction}" for id, prediction in predictions])
            return output
        except Exception as e:
            return f"Error: {str(e)}"
        
   elif title != "" and text != "":
        input_text=title+" "+text
        input_text=preprocessing_pipeline(input_text)
        input_text = input_text[:512] if len(input_text) > 512 else input_text
        result = pipe(input_text)
        label_id = result[0]["label"]
        if label_id == "LABEL_0":
            output = "Fake"
        else:
            output = "True"
        return output

   else:
        return "Error: Please enter either title and text or upload a CSV file."

demo = gr.Interface(
    fn=submit,
    inputs=[
        gr.Textbox(label="Title"),
        gr.Textbox(label="Text"),
        gr.File(label="Upload CSV file"),
    ],
    outputs="text",
    title="Fake News Detector",
    description="Enter title and text or upload a CSV file.",
    theme="primer",
)

if __name__ == "__main__":
    demo.launch()