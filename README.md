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


when reading the images we will assign 20 percent of the images to the evaluation data set 
