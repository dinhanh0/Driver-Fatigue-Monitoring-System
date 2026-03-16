Originally developed as a team capstone project

This repository contains my contributed/public version

Original private team repository not publicly accessible

I contributed on Model 1 (Resnet 18 + LSTM + NAT), located in Driver-Fatigue-Monitoring-System/src/models/resnet18_lstm.py


Here is a video I made to demonstrate model's functionality  ​
https://youtu.be/Kb0_bhePFJU 

To run model 1,
1. Do pip install requirements.txt
2. Download required datasets from kaggle and place it in Driver-Fatigue-Monitoring-System/data/
https://www.kaggle.com/datasets/esrakavalci/sust-ddd
https://www.kaggle.com/datasets/nikospetrellis/nitymed
3. Within Driver-Fatigue-Monitoring-System/src/
Run data.py to create dataset index
Run preprocess.py to preprocess raw videos into windows and labeled data
Run train.py to train the model
Run eval.py to get evaluation metrics and final results

More info can be found in our team's capstone presentation powerpoint slide
https://purdue0-my.sharepoint.com/:p:/r/personal/dinh20_purdue_edu/_layouts/15/Doc.aspx?sourcedoc=%7BA4ACC556-2438-4FE4-BD01-28CD9A9F0E44%7D&file=FinalPresentationCapstone%20-%20Copy.pptx&action=edit&mobileredirect=true
