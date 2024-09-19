# EMIT (SFID-tags) 
This is an Open Source Deeptech Blueprint

## Machine Learning on TPMS dataset

This repository contains a limited TPMS dataset and machine learning models for rapid design of EMIT based on contribution from KAIST Soft Robotics and Intelligent Materials (SRIM) Laboratory with Wiley Advanced Functional Materials journal paper [EMIT (IL-Kwon Oh et al)](https://arxiv.org/abs/1506.02640).


## Open Source Objective
EMIT needs reliability in 3D printing manufacturing and the open-source contribution is for reproduction, and reliability in engineering for its impactful application to find missing items like crashed airplanes in the ocean, sunken ships, sunken containers, or missing divers. EMIT should be open source technology to improve collaboration between AI scientists, and intelligent mechanical metamaterial scientists to develop reliable EMIT mechanical intelligence, mechanical wave chipsets, and mechanical chip-tags for mechanical transponder computing for its impactful application use.

## What is EMIT?
Encoded Mechanical Identification Tags (EMIT) are the first generation of passive sonic frequency identification (SFID) transponders unlocking the development of SFID intelligent systems empowering machines with echolocation identification sensing in acoustic telecommunication.

According to [Cambridge dictionary](https://dictionary.cambridge.org/dictionary/english/emit) or [Oxford dictionary](https://www.oxfordlearnersdictionaries.com/definition/english/emit):
The verb "emit" means to "send out light, sound, or a smell, or a gas or other substance".  EMIT (SFID-tag) emits echoes as a passive transponder when it interacts with incident sonics frequencies and sound waves. 

This sonic echolocation identification mechanism also exists in nature between two fascinating symbionts; [Borneo carnivorous pitcher plant and Borneo bat](https://commonnaturalist.com/2016/05/13/the-bats-that-live-in-carnivorous-plants/) when bat emits sound and follows its echoes changed by Borneo pitcher plant to find it for symbiotic acoustic communication in the air in Forrest. 

The first electromagnetic radio frequency identification [(RFID)-tag](https://en.wikipedia.org/wiki/Radio-frequency_identification) was invented back in 1973 by [Charles Walton ](https://en.wikipedia.org/wiki/Charles_Walton_(inventor))at MIT university unlokced RFID Transponder systems, but the first passive SFID-tag (EMIT) is invented in 2024 at KAIST university by IL-KWON OH and DANIEL SAATCHI unlocked development of SFID transponder systems. SFID-tags work based on mechanical sound waves and sonic frequencies. 

## Files
- `TPMS_BNF_data.csv`: The TPMS dispersion curve dataset used for ML training and evaluation for EMIT-design.
- `clean.py`: Script for data preprocessing for SFID-tags (EMIT) audio dataset.
- `train.py`: Script for DL training the SFID-classifer model.
- `predict.py`: Script for predicting with the trained DL model.
- `EMIT.ipynb`: Jupyter notebook documenting the entire workflow in one single file-like instruction for ML models.

## Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/danielsaatchi/emit.git
    cd your-repository
    ```
2. Set up a miniconda=4.8.3 or later version environment and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the scripts or open the Jupyter Notebook to explore the project with images and reproduction.

 
   3.1 Importing Libraries in Jupyter Notebook 
   ![image](https://github.com/danielsaatchi/EMIT/assets/47679486/86257a6f-32c9-4bf1-9873-1388557e9517)

    3.2 Importing TPMS dataset
   ![image](https://github.com/danielsaatchi/EMIT/assets/47679486/c75b88a8-a919-44c0-bbaa-ab10bb2b4a63)

    3.3 Acoustic feature extraction for sound wave and acoustic metamaterial data-driven design with machine learning
   ![image](https://github.com/danielsaatchi/EMIT/assets/47679486/5b202775-70e2-4c6b-9c91-810e13feff8b)
   
    3.4 Data Analysis: Ploting VF Variations to observe acoustic bandgaps (ABG)
    ![image](https://github.com/danielsaatchi/EMIT/assets/47679486/d66c8a7d-a9a6-46e6-b932-14325d7c92f3)
   
    3.5 Training and Sample MLP Machine Learning Prediction (Artificial Neural Network)
    ![image](https://github.com/danielsaatchi/EMIT/assets/47679486/18177b4f-2248-4802-a27d-c558d756c6ab)

    3.6 with a few features: Comparing R2 and RSME for different Machine Learning Algorithms on the same training set, validation set, and testing set 
    ![image](https://github.com/user-attachments/assets/2609381b-31a4-47bb-b9b0-78a3ba4d0647)

   3.7 with more features: Comparing R2 and RSME for different Machine Learning Algorithms on the same training set, validation set, and testing set 
    ![image](https://github.com/user-attachments/assets/95896daf-7289-4bda-b41a-e929b456e655)

   3.8 ML Generated Serial Number for EMIT (SFID-tag)
   *Note: while the dataset is complete for all types of TPMS (e.g. Diamond, Gyroid, Hexagonal, etc.) or unseen domains, the ML serial numbers can be generated with the final design metamaterial parameters

    ## Deep Learning SFID-classifier and experimental SFID-tags audio dataset 
    This section will be updated after the paper's publication and acceptance in the journal. 
        ![image](https://github.com/user-attachments/assets/357a4ce7-c191-4d4e-aa7f-625bc0999864)
    EMIT on Guitars related to the "Audio Dataset" folder for air-medium.
        ![image](https://github.com/user-attachments/assets/6198cb7d-8c38-47e3-ae06-4a654a639933)


## Link to Reference Papers
- [AFM Paper: EMIT (IL-Kwon Oh et al)](https://onlinelibrary.wiley.com/journal/16163028)
- [KAIST PhD Thesis: Daniel Saatchi (Advisor: IL-Kwon Oh)](https://drive.google.com/file/d/1n1wZJd2kUU5FUxRGdAKw6yvlHCGDI1bT/view?usp=drive_link)

## Citation for scientists and co-inventors
 - Daniel Saatchi, Myung-Joon Lee, Tushar Prashant Pandit, Manmatha Mahato, Il-Kwon Oh. 2024, Artificial Intelligence in Metamaterial Informatics for Sonic Frequency Mechanical Identification Tags. EMIT (SFID-tags). Advanced Functional Material 2024.08.09 (https://doi.org/10.48550/arXiv.2310.18257)
 - Corresponding Author: Professor IL-KWON OH (ikoh@kaist.ac.kr)
 - GitHub repository developer: DANIEL SAATCHI

## License
- KAIST: [Soft Robotics and Intelligent Materials Lab](https://srim.kaist.ac.kr/)
- CRI: [National Creative Research Initiative Center for Functionally Anatonistic Nano-Engineering](https://srim.kaist.ac.kr/)
- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python](https://img.shields.io/badge/language-Python-blue.svg)



##  Acknowledgements
This work was partially supported by the Creative Research Initiative Program (2015R1A3A2028975), funded by the National Research Foundation of Korea (NRF). This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2023-00302525).  
- The open-source dataset and codes to design EMITs (SFID-tags) are available at https://github.com/danielsaatchi/EMIT. 
- The SFID-classifer is based on customized Audio-classfier developed by https://github.com/seth814/Audio-Classification.

