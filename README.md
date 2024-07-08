# EMIT (SFID-tags) 
This is an Open Source Deeptech Blueprint

## Machine Learning on TPMS dataset

This repository contains an TPMS dataset and machine learning models for rapid design of EMIT based on contribution from KAIST Soft Robotics and Intelligent Materials (SRIM) Labratory with arXiv paper [EMIT (IL-Kwon Oh et al)](https://arxiv.org/abs/1506.02640).


## Open Source Objective
EMIT needs reliability in 3D printing manufacturing and the opensource contribution is for reproduction, and reliability in engineering for its impacful applicaiton to find missing items like crashed airplanes in ocean, sunked ship, sunked containers or missing divers. We believe EMIT should be open source technology to be improve collaboration between AI scientist, amd intelligent mechanical metamaterial scientists to develop reliable EMIT mechanical intelligence, mechanical wave chipsets and mechanical chip-tags for mechanical transponder computing for its impacful application use.

## What is EMIT?
Encoded Mechanical Identification Tags (EMIT) are first generation of passive sonic frequency identification (SFID) transponders unlocking development of SFID intelligent system empowering machines with echolocation identification sensing.

According to [Camrdige dictionary](https://dictionary.cambridge.org/dictionary/english/emit) and [Oxford Dictionary](https://www.oxfordlearnersdictionaries.com/definition/english/emit):
Verb "emit" means to "send out light, sound, or a smell, or a gas or other substance".  EMIT (SFID-tag) emits echoes as a passive transponder when interacts with incident certian sound frequency wave. 

This echololocation identificaiton mechanism also exists in nature between two facinating symbionts; [Carnivorous pitcher plant called Borneo and bat](https://commonnaturalist.com/2016/05/13/the-bats-that-live-in-carnivorous-plants/) when bat emits sound and follws its echoes changed by Borneo pitcher plant to find it for symbiotic acoustic communication in air in forrest. 

The first electromagnetic radio frequency identifcation (RFID)-tag has been invented back in 1973 by Charles Walton at MIT university, but the first SFID-tag is invented in 2024 at KAIST university by IL-KWON OH and DANIEL SAATCHI.

## Files
- `TPMS_BNF_data.csv`: The dataset used for training and evaluation.
- `clean.py`: Script for data preprocessing.
- `train.py`: Script for training the model.
- `predict.py`: Script for predicting with the trained model.
- `EMIT.ipynb`: Jupyter notebook documenting the entire workflow in one single file like instruction. 

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
3. Run the scripts or open the Jupyter notebook to explore the project with images and reproduction.

 
   3.1 Importing Libraries in Jupyter notebook 
   ![image](https://github.com/danielsaatchi/EMIT/assets/47679486/86257a6f-32c9-4bf1-9873-1388557e9517)

    3.2 Importing TPMS dataset
   ![image](https://github.com/danielsaatchi/EMIT/assets/47679486/c75b88a8-a919-44c0-bbaa-ab10bb2b4a63)

    3.3 Acoustic feature extraction for sound wave and acoustic metamaterial data-driven design with machine learning
   ![image](https://github.com/danielsaatchi/EMIT/assets/47679486/5b202775-70e2-4c6b-9c91-810e13feff8b)
   
    3.4 Data Analysis: Ploting VF Variations to observe acoustic bandgaps (ABG)
    ![image](https://github.com/danielsaatchi/EMIT/assets/47679486/d66c8a7d-a9a6-46e6-b932-14325d7c92f3)
   
    3.5 Training and Sample MLP Machine Learning Predicition (Artificial Neural Network)
    ![image](https://github.com/danielsaatchi/EMIT/assets/47679486/18177b4f-2248-4802-a27d-c558d756c6ab)

    3.6 Comparing R2 and RSME for different Machine Learning Algorithms on same training set, validation set, and testing set 
    ![image](https://github.com/danielsaatchi/EMIT/assets/47679486/f3e7c68b-1efa-4179-a2a7-df464df54eaa)

### Deep Learning SFID echoes classifer and experimental SFID-tags audio dataset 
This section will be updated after paper publication and acceptance in journal. 

### Link to Reference Papers
- [Arxiv Paper: EMIT (IL-Kwon Oh et al)](https://arxiv.org/abs/1506.02640)
- [KAIST PhD Thesis: Daniel Saatchi (Advisor: IL-Kwon Oh)](https://drive.google.com/file/d/1n1wZJd2kUU5FUxRGdAKw6yvlHCGDI1bT/view?usp=drive_link)

### Citation for scientists and co-inventors
 - Daniel Saatchi, Myung-Joon Lee, Tushar Prashant Pandit, Manmatha Mahato, Il-Kwon Oh. 2024, EMIT (SFID-tags) . Arxive 2310.18257v1 (https://doi.org/10.48550/arXiv.2310.18257)
 - Corresponding Author: Professor IL-KWON OH (ikoh@kaist.ac.kr)
 - GitHub repository developer: DANIEL SAATCHI

### License
- KAIST: [Soft Robotics and Intelligent Materials Lab](https://srim.kaist.ac.kr/)
- CRI: [National Creative Research Initiative Center for Functionally Anatonistic Nano-Engineering](https://srim.kaist.ac.kr/)
- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python](https://img.shields.io/badge/language-Python-blue.svg)



###  Acknowledgements
This work was partially supported by the Creative Research Initiative Program (2015R1A3A2028975), funded by the National Research Foundation of Korea (NRF). This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2023-00302525).  The dataset and open source code is available at https://github.com/danielsaatchi/EMIT

