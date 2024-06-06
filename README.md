# EMIT Mechanical Intelligence Design 
This is an Open Source Technology Blueprint

## Machine Learning on TPMS dataset

This repository contains an TPMS dataset and machine learning models for rapid design of EMIT based on contribution from KAIST Soft Robotics and Intelligent Materials (SRIM) Labratory with arXiv paper [EMIT Mechanical Intelligence (IL-Kwon Oh et al)](https://arxiv.org/abs/1506.02640).


## Open Source Objective
EMIT needs reliability in 3D printing manufacturing and the opensource project is for reproduction, and reliability in engineering for its impacful applicaiton to find missing items like crashed airplanes in ocean, sunked ship, sunked containers or missing divers. We believe EMIT should be open source technology to be improve collaboration between AI scientist, amd intelligent mechanical metamaterial scientists to develop reliable EMIT mechanical intelligence and mechanical wave chipsets for mechanical computers for its impacful application use.

## What is EMIT?
Encoded Mechanical Intelligence Tag (EMIT)


## Files
- `TPMS_BNF_data.csv`: The dataset used for training and evaluation.
- `preprocess.py`: Script for data preprocessing.
- `train_model.py`: Script for training the machine learning model.
- `evaluate_model.py`: Script for evaluating the trained model.
- `notebook.ipynb`: Jupyter notebook documenting the entire workflow in one single file like instruction. 

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



### Link to Reference Papers
- [Arxiv Paper: EMIT Mechanical Intelligence Design (IL-Kwon Oh et al)](https://arxiv.org/abs/1506.02640)
- [KAIST PhD Thesis: Daniel Saatchi (Advisor: IL-Kwon Oh)](https://drive.google.com/file/d/1n1wZJd2kUU5FUxRGdAKw6yvlHCGDI1bT/view?usp=drive_link)

### Citation for Scientists
 Daniel Saatchi, Myung-Joon Lee, Saewoong Oh, Pandit Tushar Prashant, Ji-Seok Kim, Hyunjoon Yoo, Manmatha Mahato, Il-Kwon Oh (2024). EMIT Mechanical Intelligence Design. Arxive 2310.18257v1 (https://doi.org/10.48550/arXiv.2310.18257)

### License
- KAIST: [Soft Robotics and Intelligent Materials Lab](https://srim.kaist.ac.kr/)
- CRI: [National Creative Research Initiative Center for Functionally Anatonistic Nano-Engineering](https://srim.kaist.ac.kr/)
- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python](https://img.shields.io/badge/language-Python-blue.svg)



###  Acknowledgements
This work was partially supported by the Creative Research Initiative Program (2015R1A3A2028975), funded by the National Research Foundation of Korea (NRF). This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2023-00302525).  The dataset and open source code is available at https://github.com/danielsaatchi/EMIT

