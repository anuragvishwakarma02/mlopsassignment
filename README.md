# RoBERTa Goodreads Genre Classifier

# Roll No - G25AIT2017
MLOps | PGD AI Program | IIT Jodhpur - Assignment 2

This project fine-tunes a RoBERTa model on Goodreads book reviews from the UCSD Book Graph dataset to classify reviews into 8 genres. Experiment runs are tracked with Weights and Biases and the final trained model is uploaded to Hugging Face Hub.

The 8 genres covered are: poetry, children, comics and graphic, fantasy and paranormal, history and biography, mystery/thriller/crime, romance, and young adult.

## Important Configuration

All the main settings are in Cell 14. Update them as per need and platform:

- MODEL_CLS is set to RobertaForSequenceClassification. ( Uncomment the below lines in cell 14 to run Distilbert)
- TOKEN_CLS is set to RobertaTokenizer, the matching tokenizer for roberta
- model_name is roberta-base, the pre-trained checkpoint downloaded from Hugging Face
<!-- # MODEL_CLS = DistilBertForSequenceClassification
# TOKEN_CLS = DistilBertTokenizerFast
# model_name = 'distilbert-base-cased' -->

MODEL_CLS = RobertaForSequenceClassification
TOKEN_CLS = RobertaTokenizer
model_name = 'roberta-base'

- device_name is cuda for Kaggle GPU( T4 x2). Change it to mps if you are on a Mac M-series chip, or cpu if you have no GPU
- max_length is 512, which is the maximum number of tokens roberta can handle per input
- ENABLE_WANDB set to True will log all metrics to your W&B dashboard automatically
- ENABLE_HF set to True will push the model and tokenizer to Hugging Face after training
- HF_REPO is anuragvishwakarma02/roberta-base-goodreads-genres, the destination repository on Hugging Face
- cached_model_directory_name is roberta-base-reviews-genres, where the model is saved locally

For training, the hyperparameters used are 3 epochs, a batch size of 10 for training and 16 for evaluation, learning rate of 5e-5, 100 warmup steps, and weight decay of 0.01. The model is evaluated and saved at the end of each epoch and the best checkpoint is loaded at the end.

For data, we stream the first 10,000 reviews per genre, randomly sample 2,000 from those, then use 1,000 per genre split as 800 train and 200 test. That gives 6,400 training reviews and 1,600 test reviews in total.


## Steps to Run

You need to run this on Kaggle because it has a free GPU and local CPU training is too slow for this dataset.

1. Open Kaggle, go to Notebooks, create a new notebook, then use File > Import Notebook to upload the file g25ait2017-classifying-goodreads.ipynb
2. Go to Settings and set the Accelerator to GPU T4 x2
3. Also in Settings, turn Internet on (needed to download the dataset and push the model to Hugging Face)
4. Go to Add-ons > Secrets and add two secrets: WANDB_API_KEY (get it from wandb.ai/settings) and HF_TOKEN 
5. Click Run All and let it finish

The whole thing takes around 15 to 20 minutes on a T4 GPU for 3 epochs over 6400 training examples.




## Hugging Face Model

The fine-tuned model is available at:
https://huggingface.co/anuragvishwakarma02/roberta-base-goodreads-genres

To use it in your own code:

```python
from transformers import RobertaForSequenceClassification, RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("anuragvishwakarma02/roberta-base-goodreads-genres")
model = RobertaForSequenceClassification.from_pretrained("anuragvishwakarma02/roberta-base-goodreads-genres")
```



## Why RoBERTa Over DistilBERT

I have both models imported in the notebook but RoBERTa was chosen as the final model for a few reasons.

DistilBERT is a distilled version of BERT, meaning it was trained to mimic a larger model while being smaller and faster. It uses about 66 million parameters and runs faster than full BERT. It is a good choice when you need speed or have limited compute.

Roberta on the other hand was trained from scratch on a much larger dataset, about 10 times more data than BERT used, including CC-News, OpenWebText, and Stories. 
Roberta has around 125 million parameters.

Roberta consistently scores higher than DistilBERT on text classification benchmarks. For this assignment we are training once on a Kaggle GPU so inference speed was not a concern. The accuracy improvement from using Roberta is worth the extra compute time. 
If this were a real-time production API where latency mattered, DistilBERT would be the better pick.

Results for roberta
                            precision    recall  f1-score   support

              children       0.70      0.72      0.71       200
        comics_graphic       0.83      0.81      0.82       200
    fantasy_paranormal       0.39      0.40      0.40       200
     history_biography       0.59      0.60      0.60       200
     mystery_thriller_crime  0.55      0.58      0.57       200
                poetry       0.87      0.74      0.80       200
               romance       0.58      0.63      0.60       200
           young_adult       0.44      0.41      0.42       200

              accuracy                           0.61      1600
             macro avg       0.62      0.61      0.62      1600
          weighted avg       0.62      0.61      0.62      1600

## Results comparision

Both models were evaluated on the same 1,600 test reviews (200 per genre). Here is how they compared.

Overall accuracy: DistilBERT got 54.0% and Roberta got 61.4%. That is a 7.4 percentage point improvement just from switching the model, with no other changes to the pipeline.

Per-genre F1 scores:

    Genre                   DistilBERT F1    Roberta F1    
    children                0.665            0.711         
    comics and graphic      0.647            0.818         
    fantasy and paranormal  0.399            0.397         
    history and biography   0.559            0.599         
    mystery thriller crime  0.495            0.567         
    poetry                  0.691            0.801         
    romance                 0.512            0.603         
    young adult             0.330            0.425         

Macro average F1: DistilBERT 0.537, Roberta 0.615

Roberta improved on every genre except fantasy and paranormal where both models essentially tied around 0.40. The biggest gains were in comics and graphic (+0.171) and poetry (+0.110). Both models struggled most with young adult and fantasy, likely because those genres have a lot of overlap with romance and general fiction in the way reviewers write about them.

## Kaggle Link
https://www.kaggle.com/code/anuragg25ait2017/g25ait2017-classifying-goodreads

## W&B Experiment Tracking
Report link https://api.wandb.ai/links/g25ait2017-prom-iit-rajasthan/20at8lr8

<s> Dashboard: https://wandb.ai/g25ait2017-prom-iit-rajasthan/mlops-assignment2 </s>


The Trainer logs train loss, eval loss, accuracy, F1, precision, and recall automatically at each epoch.
After evaluation, final metrics are logged separately under the final/ prefix. A JSON classification report is also saved and uploaded as a versioned W&B Artifact called eval-report.



<img width="2056" height="1290" alt="image" src="https://github.com/user-attachments/assets/6ddbec02-74e7-4559-bbf5-019bfaaec02a" />

<img width="2056" height="1290" alt="image" src="https://github.com/user-attachments/assets/f71153f2-0f39-4bab-8b09-705e4c2c69c4" />

<img width="2056" height="1290" alt="image" src="https://github.com/user-attachments/assets/aa766351-f1c2-495c-a919-555f2bcb0850" />


