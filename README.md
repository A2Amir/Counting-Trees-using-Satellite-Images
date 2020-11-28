# Counting Trees through Satellite Images

## 1. Introduction:

Counting trees manually from satellite images above is a very tedious task. Fortunately, there are several techniques for automating tree counting. Morphological operations and classical segmentation algorithms like the watershed algorithm have been applied to tree counting with limited success so far. However, in case of dense areas, the trees are more densely packed and the crowns of the tress often overlap. These areas probably show different forest characteristics, such as differences in crown structure, species diversity, openness of tree crowns. This makes the problem more difficult. Therefore the tree counting algorithm has to be more robust and intelligent. This is where deep learning comes into play.

This study investigates the aspect of **Localization and Counting of Trees** to create their inventory for incoming and outgoing trees, which (due to extensive felling) are only documented and recorded in the tree register during the annual tree inspections. 



## 2. Dataset and Processing:

Satellite images are usually very large and have more than three channels. Our dataset  consist of satellite images (848 × 837 pixels and eight channel) and labeled masks ( has 848 × 837 pixels and five channel) which are hand label by the analysts with image labeling tools to present:

<b>
  
1. Buildings
  
2. Roads and Tracks

3. Tress

4. Crops

5. Water
</b>

Below you see one of the satellite images and the corresponding labels:


<p align="center">
  <img src='./imgs/2.png' alt="the satellite images and the corresponding labels" width="1300" height="200" >
 </p>


In order to create training  and validation  dataset, the steps below were implemented:

1. When reading the satellie images and it's corresponding lables,  20 percent of each images and labels were assigned  to the evaluation data set.
2. Once the training dataset and the validation dataset are created, a random window with a predefined size moves over the images and labels of the training dataset and the validation dataset to create the predefined number of patches.  For example, with a window size of 160 and 4000 patches for the training data set, we have a shape of (4000, 160,  160, 8) for the training images and a shape of (4000, 160, 160, 5) for the training labels.
3. Since we will focus on counting the trees in this study, the four other channels of labels, namely buildings, roads and tracks, crops and water will be removed. i.e., the form of the training labels explained above will be (4000, 160, 160, 1).



