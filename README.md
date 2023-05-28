# Trash Detection Model 

A neural network model that will be able to accurately detect and label trash given a certain image or a live camera feed using bounding boxes! This could be used to address multiple SDG goals, which include but not limited to the following:

SDG 11: Sustainable Cities and Communities - By detecting and removing labelled trash from public spaces, this technology can help create cleaner and more sustainable urban environments.

SDG 12: Responsible Consumption and Production - The neural network model can help reduce waste and encourage responsible consumption by identifying trash and waste products that can be recycled or repurposed.

SDG 13: Climate Action - By reducing the amount of waste that ends up in landfills, this technology can help reduce greenhouse gas emissions and mitigate the impacts of climate change.

SDG 14: Life Below Water - By preventing trash and plastic waste from entering waterways, the technology can help protect marine ecosystems and preserve biodiversity.

SDG 15: Life on Land - By removing litter and waste from natural environments, the model can help protect natural habitats and prevent the loss of biodiversity.

DEMO VIDEO: https://youtu.be/Ka9styd9CAI


Necessary Packages / Requirements
    All ultralytics reqiurements (for running the YOLOv8 model) https://github.com/ultralytics/ultralytics/blob/main/requirements.txt
    
    
 Model Details:
    Data sources used for training: 
        Data used for VGG19 Transfer learning: https://www.kaggle.com/datasets/sovitrath/underwater-trash-detection-icra
        Data used for YOLOv8 model training:   https://universe.roboflow.com/mohamed-traore-2ekkp/taco-trash-annotations-in-context/dataset/16
   
    
DOWNLOAD THE MODEL HERE:

https://drive.google.com/file/d/1RcG_nSd4TcYYaOV8TuIh_GCrnpnLg_VF/view?usp=sharing

Currently only the YOLO model can be used for real time webcam detection.

How to Run:
Download "thecode.py"
After downloading the YOLO model, figure out what directory you have downloaded it to.
Replace "DIRECTORY OF WHERE YOU DOWNLOADED MODEL" with the directory of where your model is. (for example Users/yourusername/Downloads/modelname)
Run the code, it should use your webcam for the source.


Future Work Ideas / Next Steps:

Additional optional data preprocessing steps for murky water pictures and foggy pictures. (Maybe nightvision for images in the dark)
Somehow fix detecting faces as 'trash'.
Optimize code for better frames per second
Audio after detection
Mount on an actual robot


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

05/19/23-05/23/23
- Trained YOLOv8 model with new dataset
- Added live camera feed detection
