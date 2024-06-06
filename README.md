# EMIT Mechanical Intelligence Design (Open Source)
## Machine Learning on TPMS dataset

This repository contains an TPMS dataset and machine learning models based on contribution from KAIST Soft Robotics and Intelligent Materials (SRIM) Labratory with arXiv paper [EMIT Mechanical Intelligence (IL-Kwon Oh et al)](https://arxiv.org/abs/1506.02640).


## Open Source Scope
EMIT needs reliability in 3D printing manufacturing and the opensource project is for reproduction, and reliability in engineering for its impacful applicaiton to find missing items like crashed airplanes in ocean, sunked ship, sunked containers or missing divers. We believe EMIT should be open source technology to be improve collaboration between AI scientist amd intelligent mechanical metamaterial scientists to develop reliable EMIT mechanical intelligence for its impacful application use.

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

 
   #### 3.1 Importing Libraries in Jupyter notebook 
   ![image](https://github.com/danielsaatchi/EMIT/assets/47679486/86257a6f-32c9-4bf1-9873-1388557e9517)


### Link to Reference Papers
- [Arxiv Paper: EMIT Mechanical Intelligence Design (IL-Kwon Oh et al)](https://arxiv.org/abs/1506.02640)
- [KAIST PhD Thesis: Daniel Saatchi (Advisor: IL-Kwon Oh)](https://drive.google.com/file/d/1n1wZJd2kUU5FUxRGdAKw6yvlHCGDI1bT/view?usp=drive_link)

### Citation for Scientists
 Daniel Saatchi, Myung-Joon Lee, Saewoong Oh, Pandit Tushar Prashant, Ji-Seok Kim, Hyunjoon Yoo, Manmatha Mahato, Il-Kwon Oh (2024). EMIT Mechanical Intelligence Design. Arxive 2310.18257v1 (https://doi.org/10.48550/arXiv.2310.18257)

###  Licensed by KAIST and CRI
- [Soft Robotics and Intelligent Materials Lab](https://srim.kaist.ac.kr/)
- [National Creative Research Initiative Center for Functionally Anatonistic Nano-Engineering](https://srim.kaist.ac.kr/)

![Python](https://img.shields.io/badge/language-Python-blue.svg)

###  Acknowledgements
This work was partially supported by the Creative Research Initiative Program (2015R1A3A2028975), funded by the National Research Foundation of Korea (NRF). This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2023-00302525).  The dataset and open source code is available at https://github.com/danielsaatchi/EMIT

