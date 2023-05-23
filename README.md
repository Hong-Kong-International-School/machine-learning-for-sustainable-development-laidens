#Trash Detection Model 

A neural network model that will be able to accurately detect and label trash given a certain image or a live camera feed using bounding boxes! This would be used in trash collection robots that could autonomously pick up trash, if we were ever able to develop this kind of robot.

Necessary Packages / Requirements
    All ultralytics reqiurements (for running the YOLOv8 model) https://github.com/ultralytics/ultralytics/blob/main/requirements.txt
    
    How to Install: Download necessary packages shown at the top of the 'thecode.py' file. 
    How to Use: Go 
    
 Model Details:
    Data sources used for training: 
        Data used for VGG19 Transfer learning: https://www.kaggle.com/datasets/sovitrath/underwater-trash-detection-icra
        Data used for YOLOv8 model training:   https://universe.roboflow.com/mohamed-traore-2ekkp/taco-trash-annotations-in-context/dataset/16
        Model performance analysis: 
            VGG19 Transfer learning model performance: 
    
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
