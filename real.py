import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

num_classes = 5

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))
my_new_model.layers[0].trainable = False

my_new_model.summary()

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

train_dir = '../input/dinosaur-data/dino-data/train'
val_dir = '../input/dinosaur-data/dino-data/test'

batch_size = 10
img_size = (224, 224)


from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
)


train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    seed=42
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False,
    seed=42
)


images, labels = val_generator.next()
random_indices = np.random.choice(batch_size, size=9, replace=False)
random_images = images[random_indices]
random_labels = np.argmax(labels[random_indices], axis=1)

# Define the plot grid and size
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

# Loop over the 9 images and plot them with data augmentation
for i in range(9):
    row = i // 3
    col = i % 3
    ax[row, col].imshow(random_images[i])
    ax[row, col].axis('off')

plt.tight_layout()
plt.show()


CLASS_LABELS  = ['Ankylosarus', 'Brontosaurus', 'Pterodactyl', 'T-Rex', 'Triceratops']

fig = px.bar(x = CLASS_LABELS,
             y = [list(train_generator.classes).count(i) for i in np.unique(train_generator.classes)] , 
             color = np.unique(train_generator.classes) ,
             color_continuous_scale="Emrld") 
fig.update_xaxes(title="Type of Dinosaur")
fig.update_yaxes(title = "Number of Images")
fig.update_layout(showlegend = True,
    title = {
        'text': 'Train Data Distribution ',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()

my_new_model.fit(
        train_generator,
        steps_per_epoch=3,
        epochs = 7,
        validation_data=val_generator,
        validation_steps=1)

my_new_model.evaluate(val_generator)
preds = my_new_model.predict(val_generator)
y_preds = np.argmax(preds , axis = 1 )
y_test = np.array(val_generator.labels)

cm_data = confusion_matrix(y_test , y_preds)
cm = pd.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')