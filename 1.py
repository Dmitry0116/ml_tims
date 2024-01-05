from transformers import pipeline

clf = pipeline(
    task = 'text-classification', 
    model = 'cointegrated/rubert-tiny2-cedr-emotion-detection')

text = ['УРАААААААА',
        'Какая гадость эта ваша ...!']        

print(clf(text))
