# Coral image dataset for coral conditions monitoring in the Indo-Pacific
Dataset and tabular data for coral condition classification paper: Deep learning for multi-label classification of coral conditions in the
Indo-Pacific via underwater photogrammetry.
## Project
### Project title
The Koh Tao Coral Condition Survey Project was initiated by Jun Sasaki’s laboratory and implemented in collaboration with the New Heaven Reef Conservation NPO, Koh Tao, Thailand.
### Project description
Given the Koh Tao Island case context, this research aims to apply state-of-the-art deep learning models to devise a new multi-label classification model for coral reef conservation, mainly aiming at revolutionizing the current monitoring program by integrating underwater images and citizen science text data. Coral images were collected by divers holding an Olympus TG-6 with waterproof housing. Geographical parameters were collected as metadata and saved as tabular data of the supplementary materials.
## Dataset
### Dataset description
The project covered most of the coral habitats surrounding Koh Tao, selected based on the weather conditions and characteristics of the habitats. The dataset consists of (1) the original, uncropped 3000*4000 pixels images; (2) cropped, 512*512 pixels image patches for the training and test of the model; (3) classification labels annotated by human experts; and (4) metadata of each field survey.
![Study area](https://github.com/XL-SHAO/CoralConditionDataset/assets/117028875/b3a5df33-c3b0-4b9f-8b1c-09970549adf4)
### Dataset organization
There are 23,965 image patches generated from nine times of field surveys. At each site, the sampling strategy is as follows:
![image](https://github.com/XL-SHAO/CoralConditionDataset/assets/117028875/9f417cd5-7aea-4cb7-b18b-d28706c904e9)
All the image tiles are in JPG format. The images are organized based on the survey they were collected from. For example, the folder ‘20230804_CBK’ is named after the survey date_abbreviation of the site. The full name of each site can be found in the ‘surveys_metadata’ in the tabular_data folder.
### Image naming conventions
The name of each image consists of five parts separated by underscores: the three-letter site code of the sampling site (e.g., ALK=Ao Leuk, CBK=Chalok Baan Kao Bay), the unique four-digit survey number, the two-digit transect number, the date of the survey formatted as YYYYMMDD, the four-digit image number, and the number of the image patch. Examples of the images are as follows:
CBK_0001_11_20230805_0001_12, TTB_0002_00_20230815_0002_02
### Dataset content
**1.Metadata:** (surveys_metadata.csv) this file provides additional information about the images, and it is organized by the survey id, which refers to every single time of the field observation. The parameters contained in this table are as follows:
* surveyid: the unique four-digit code representing each observation. For instance, the first field observation was conducted on August 4th, 2023, at Chalok Baan Kao Bay; all the images from this observation will be named with survey id 0001.
* transectid: the two-digit code representing the transect lines. The first digit identifies if the transect is permanent (1-permanent, 0-temporal), and the second digit reflects if the transect is the 1-shallow line (2-4m depth) or 2-deep line (9-10m depth). Additionally, 00 stands for images collected randomly without setting up transects.
* survey_date: in YYYYMMDD form, reflecting the date the survey is conducted.
* site: the three-letter site identification code representing which diving site those data were collected from.
* folder_name: name of the folder where the images were saved.
* lat_start, lng_start: the latitude and longitude of the starting point of this survey.
* camera: camera used to capture the images.
* depth: average depth of this survey.
* temp: average temperature of this survey.

**2.Labelset of the annotation:** (labelsets.csv) this table contains the set of possible labels that can be assigned to the objects within the images. It includes four labels reflecting the coral conditions and four for stressors.

**3.Annotations:** (annotations.csv) annotation file contains the image patch id and its corresponding classification label, which is used for model training and image classification. This annotation was labeled by human experts in marine ecology and coral conservation.
# Citation
If this dataset contributes to your research, please consider citing our paper:
```LaTeX
@article{shao2024coral,
  title={Deep learning for multi-label classification of coral conditions in the Indo-Pacific via underwater photogrammetry},
  author={Shao, Xinlei and Chen, Hongruixuan and Magson, Kirsty and Wang, Jiaqi and Song, Jian and Chen, Jundong and Sasaki, Jun},
  journal={Aquatic Conservation: Marine and Freshwater Ecosystems},
  year={2024}
}
```
# Q & A
For any questions, please [contact us.](mailto:yuishaoxinlei@gmail.com)
