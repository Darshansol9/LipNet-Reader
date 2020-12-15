# LipNet-Reader
Aim to detect lip movements from user video streams without audio and convert into text 

1) Install the required libraries
```
pip install requirements.txt
```
2) Run the dataloader script
``` shell dataloader.sh ```
> Note: Change the paths accordingly, for simplicity look for *path_to_save / lookup path* and replace with your intended path.
4) Place all the scripts of training_scripts and video_processing folder inside one folder and start executing 
    * ```python image_encodings.py```
        This will take around 28-30 hrs depending upon the resources you have.
        Upon execution you should have **image_encodings.pikcle,text_sentences.pikcle,ignore_files.ext,corpus.txt** getting generated
    * ```python model_to_train.py``` - This took around 16 hrs of training on 2 distributed Tesla V100 32 GB memory.
    * If you do not have that much resources, you just to need to sample the data from X_train_full,X_test_full,X_val_full.txt and re-run the steps.
5) If someone want to use the pre-trained model, one can find the model architecture and saved weights inside *training_scripts/saved_model* This model is saved on 500 epochs with early stopping being in place.


See the complete detailed report of our work './report.pdf'

