# Underwater Trash Detection Model

System Details:
    This model will be able to accurately detect trash in an underwater environment. This kind of model would be used in ocean trash collection robots that     autonomously pick up trash and remove it from the water. 
    
    How to Install: Download necessary packages shown at the top of the 'thecode.py' file. 
    How to Use: Go 
    
 Model Details:
    Data sources used for training: https://www.kaggle.com/datasets/sovitrath/underwater-trash-detection-icra
    Model performance analysis: placeholder
    
 Development overview:
    placeholder


##  Logs

04/17/23
- Started Documenting logs for significant progress and updates
- Ideas for project: 
    SDG 6 (Clean Water and Sanitation) - Evaluate aspects of water using computer vision and CNN's to classify a picture of water as 'drinkable' or not.
    SDG 14 (Life Under Water) - Use CNN's to detect plastic trash/objects in the water.
- Current Problems:
    Downloading certain packages like tensorflow do not work. Tensorflow installed on computer. Needs further troubleshooting
- Found multiple datasets containing photos of trash underwater. However, I can't upload files larger than 25MB onto github. I will most likely post the hyperlink to where I got the dataset instead. Additionally, the majority of these datasets also have extraneous files such as .txt and .xml files that I don't need. A lot of photos look almost identical to each other too, so I will have to do some preprocessing.

04/21/23
- Fixed Tensorflow issue
- Made commit comments clearer (lol)
- Started creating a model. I started the project on kaggle for now because I am still having some issues with downloading packages on my macbook. Will transfer onto github soon

05/05/23
- Updated README

05/06/23 - 05/19/23
- Trained Model with VGG19 Transfer learning (planning on also trying a YOLO model)
- Added appropiate performance visualization such as a loss function graph
- Added examples of the model applied on validation data.