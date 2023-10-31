from youtube_transcript_api import YouTubeTranscriptApi
import requests
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from langchain.output_parsers import CommaSeparatedListOutputParser
import streamlit as st
import re
from reportlab.pdfgen import canvas

pdf = canvas.Canvas("questions.pdf")

# Add a title
pdf.setTitle("Questions")

def summarymaker(text):
        stopWords = set(stopwords.words("english"))
        words = word_tokenize(text)
        freqTable = dict() 
        for word in words:
            word = word.lower()
            if word in stopWords:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1
        
        sentences = sent_tokenize(text)
        sentenceValue = dict()
        
        for sentence in sentences:
            for word, freq in freqTable.items():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq
        sumValues = 0
        for sentence in sentenceValue:
            sumValues += sentenceValue[sentence]
        
        average = int(sumValues / len(sentenceValue))

        summary = ''
        for sentence in sentences:
            if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
                summary += " " + sentence
        out_file=open("summary_out.txt","w+")
        out_file.write(summary)
        return summary

api="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"

llm1 = OpenAI(openai_api_key=api, temperature=0.7)
output_parser = CommaSeparatedListOutputParser()
template1 = """You generate a list of 5 multiple choice questions, ONLY 3 possible answers based on video transcript that we send to you 
Question: Generate 5 multiple choice questions, ONLY 3 possible answers from the following transcript: {text}
Answer: 5 multiple choice questions, ONLY 3 possible answers in json. The question-answer format should always be the same.
This is format the questions should be in for each question and its 3 possible answers:
[Question Number(so 1, 2, 3 , 4 or 5)][A Period][Empty Space][Actual Question]
[Answer Choice Letter][A period][Empty Space][Answer Choice Description]
[Answer Choice Letter][A period][Empty Space][Answer Choice Description]
[Answer Choice Letter][A period][Empty Space][Answer Choice Description]
[Answer Choice Letter][A period][Empty Space][Answer Choice Description]
[Answer Choice Letter][A period][Empty Space][Answer Choice Description]
[Empty New Line]
"""

prompt_template1 = PromptTemplate(input_variables=["text"], template=template1, output_parser=output_parser)
answer_chain1 = LLMChain(llm=llm1, prompt=prompt_template1)

st.set_page_config(
    page_title = "Studymint Dashboard",
    layout="wide"
)

st.title("Studymint 🌿🍃")
st.caption("Made by Shervin A.")

link = st.text_input("Enter YouTube Link to generate answers to questions")
submit = st.button("Submit")

if submit:
    st.write(link)
    with st.spinner('Generating your questions: '):
        video_id = re.search(r"^(?:https?:\/\/)?(?:www\.)?(?:m\.)?youtu(?:\.be|be\.com)\/(?:watch\?v=|embed\/|v\/|shorts\/)?(?P<video_id>[\w-]{11})(?:\S+)?$", link).group("video_id")
        json = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = summarymaker(' '.join(list(map(lambda word: word['text'], json))))
        st.write(f"Got transcript for link: {link}")

        questions = answer_chain1.run(transcript)
        
    st.write(transcript)
    st.write(questions.splitlines())
    for i, question in enumerate(questions.splitlines()):
      pdf.drawString(10, 800 - i * 15, question)

    pdf.save()


    st.balloons()
    st.success('Questions PDF Sucessfully generated!', icon="✅")

    with open("questions.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()
    st.download_button(label="Download Questions",
                    data=PDFbyte,
                    file_name="test.pdf",
                    mime='application/octet-stream')
