# AI4D iCompass Social Media Sentiment Analysis for Tunisian Arabizi

## Brief Description

TUNIZI is the first 100% Tunisian Arabizi sentiment analysis dataset, developed as part of AI4D’s ongoing NLP project for African languages. Tunisian Arabizi is the representation of the Tunisian dialect written in Latin characters and numbers rather than Arabic letters.   
The objective of this challenge is to build a sentiment analysis classifier for the Tunisian Arabizi Dialect.   
For more information about this challenge, have a look on [Zindi](https://zindi.africa/competitions/ai4d-icompass-social-media-sentiment-analysis-for-tunisian-arabizi).   

## Repo Structure

|---- nlp (package)  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- . . .   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- *{module}.py*   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- . . .   
| \
|---- data (placeholder for raw and preprocessed data)  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- Train.csv   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- Test.csv  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- SampleSubmission.csv   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- . . .  \
| \
|---- notebooks  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- AI4D_Processing.ipynb  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- AI4D_rzA27Luehf.ipynb  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- AI4D_AH7LwUXCvT.ipynb  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- AI4D_10WwJdQcXs.ipynb  
|\
|---- submissions (auto-generated)  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- *.csv   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- *.csv   
|\
|---- setup.py
|\
|---- Readme.md   

PS: This isn't the definitive structure. During the code execution, new directories will be created.

## How to run the code

### Steps

```
# 1. Make sure to follow the repo structure
# 2. Run 'pip install ./'
# 3. Run 'notebooks/AI4D_Processing.ipynb'
# 4. Run 'notebooks/AI4D_rzA27Luehf.ipynb', 'notebooks/AI4D_AH7LwUXCvT.ipynb', 'notebooks/AI4D_10WwJdQcXs.ipynb'
# 5. Run 'python blend.py '
```

### Expectations

To make sure that everything is working smoothly, here is what to expect from above (steps):

```
# 1. 
# 2. This step installs the nlp package
# 3. After this step, verify that 'data/{TrainNormalized.csv, TestNormalized}.csv' exist
# 4. Directory 'submissions/' will be added to the repo structure and contain '{multi-dialect-bert-base-arabic*, bert-multilingual-cased*, roberta-base*}.csv'.
# 5. Performs a simple weight-blend, then creates 'submissions/final_submission.csv' which is the final submission file.
```
## [On the Leaderboard](https://zindi.africa/competitions/ai4d-icompass-social-media-sentiment-analysis-for-tunisian-arabizi/leaderboard)

Look for : [**Muhamed_Tuo**](https://zindi.africa/users/Muhamed_Tuo) <br>
Rank : 9th/312   
Accuracy Score: 0.8362(Private) - 0.8394(Public)   

## Authors

<div align='center'>

| Name           |                     Zindi ID                     |                  Github ID               |
|----------------|--------------------------------------------------|------------------------------------------|
|Muhamed TUO     |[@Muhamed_Tuo](https://zindi.africa/users/Muhamed_Tuo)  |[@NazarioR9](https://github.com/NazarioR9)|


</div>
