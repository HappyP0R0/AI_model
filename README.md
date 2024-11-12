# AI_model

Bachelors thesis project about driver attention clasification using deep learning

## Dataset
Project is using **"100-Drivers"** dataset
It can be found at: [https://100-driver.github.io]([url](https://100-driver.github.io))

Experiments where done using:
- Images taken in daylight
- Images taken in Night time

Runs that used images taken in daylight for training and testing were marked as D1ToD1
Runs that used images taken in daylight for training and images taken at night time for testing were marked as D1ToN1

We used images taken at night time to check stats on images that were completely different from training data.

## Model
Experiments where done using the model created by combining:
- EfficientNetB0 and LSTM (marked as D1ToD1NoPooling or D1ToN1NoPooling)
- EfficientNetB0, LSTM and adding Pooling at the end (marked as D1ToD1Pooling or D1ToN1Pooling)

## Stats
From D1ToD1Pooling stats we can see that model with Pooling gets higher stats.
- Accuracy **81.91%**  
- F1 Score of **81.50**
- Loss Function Value **0.0546** 

Experiments with Night time photos show similar results, but with lower scores: 
- Accuracy: **51.06%** 
- F1-Score: **49.98** 
- Loss Function: **0.1552** 

The experiment with night images took longer, so it got through only 5 steps (epoches) in 3 days.

### D1ToD1NoPooling vs D1ToD1Pooling
![Accuracy Graph](https://github.com/user-attachments/assets/97496852-585e-4715-8351-26e228010af4)

![F1 Score Graph](https://github.com/user-attachments/assets/9eb6d219-89c1-4cd0-a8a9-0464ed6217b7)

![Loss Function Graph](https://github.com/user-attachments/assets/e0476a32-89f2-4c91-86ab-7443a01ba0c1)

### D1ToN1Pooling vs D1ToN1NoPooling

![AccuracyGraphD1N1](https://github.com/user-attachments/assets/3e96b5dd-e342-4206-9a44-f770ac836329)

![F1ScoreGraphD1N1](https://github.com/user-attachments/assets/64cdbb21-b060-4c3d-a2ed-dbd28df9cfd0)

![LossFunctionGraphD1N1](https://github.com/user-attachments/assets/248863f9-8bf5-4f13-a495-c4886c2a96b8)
