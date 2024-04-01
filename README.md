# **MELT: Mutual Enhancement of Long-Tailed User and Item for Sequential Recommendation**

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://sigir.org/sigir2023/" alt="Conference">
        <img src="https://img.shields.io/badge/SIGIR'23-lightgray" /></a>
</p>

The official source code for [MELT: Mutual Enhancement of Long-Tailed User and Item for Sequential Recommendation paper](http://arxiv.org/abs/2304.08382), accepted at SIGIR(full paper) 2023.

## **Overview**  

### Overall Framework  
<img src="figure/Main.png" width="300">

### Abstract   

The long-tailed problem is a long-standing challenge in Sequential
Recommender Systems (SRS) in which the problem exists in terms of both users and items. While many existing studies address the long-tailed problem in SRS, they only focus on either the user or item perspective. However, we discover that the long-tailed user and item problems exist at the same time, and considering only either one of them leads to sub-optimal performance of the other one. In this paper, we propose a novel framework for SRS, called Mutual Enhancement of Long-Tailed user and item (MELT), that jointly alleviates the long-tailed problem in the perspectives of both
users and items. MELT consists of bilateral branches each of which is responsible for long-tailed users and items, respectively, and the branches are trained to mutually enhance each other, which is
trained effectively by a curriculum learning-based training. MELT is model-agnostic in that it can be seamlessly integrated with existing SRS models. Extensive experiments on eight datasets demonstrate the benefit of alleviating the long-tailed problems in terms of both users and items even without sacrificing the performance of head users and items, which has not been achieved by existing methods.
To the best of our knowledge, MELT is the first work that jointly alleviates the long-tailed user and item problems in SRS.


## **Data Preprocess**  

### **1. Download the raw datasets (i.e., ratings only) in the following links**  

* [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html): Download the raw datasets instead of "5-core" datasets.

* [Behance](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/behance/) : You can download the "Behance_appreciate_1M.gz" in the data explorer.

* [Foursquare](https://archive.org/details/201309_foursquare_dataset_umn)   
 

 
### **2. Then, put the raw datasets in the *raw_dataset* directory** 

### **3. Preprocess the datasets using **preprocess.ipynb** file**

We follow the same-preprocessing strategy with [SASRec](https://github.com/kang205/SASRec/blob/master/data/DataProcessing.py).

## **Library Versions**

* Python: 3.9.12  
* Pytorch : 1.10  
* Numpy: 1.21.2  
* Pandas: 1.3.4  

We upload the **environment.yaml** file to directly install the required packages.

``` python  
conda env create --file environment.yaml
``` 

## **Training**



### 1. Prepare the trained backbone (e.g., SASRec, FMLP) model.

We share the pretrained backbone encoder.  
You can donwload the pretrained model and put it on *save_model/{DATA_NAME}* directory.

* [SASRec](https://drive.google.com/drive/folders/1SKpdN_mAyMJgLTLSbqJOi3C9b8zm9Gbp?usp=sharing)

* [FMLP](https://drive.google.com/drive/folders/1D-dWuWKQB1VOwC91w26jjD1CvXqs2qx9?usp=sharing)

Or you can explicitly train the SASRec or FMLP model with the following commands.


#### SASRec  

``` python  
# In the shell code, please change the 'data' variable 
bash shell/SASRec/train_amazon.sh # Dataset: Clothing, Sports, Beauty, Grocery, Automotive, Music
```  

#### FMLP 

``` python  
# In the shell code, please change the 'data' variable 
bash shell/FMLP/train_amazon.sh # Dataset: Clothing, Sports, Beauty, Grocery, Automotive, Music 
```  



### 2. Train the MELT framework.

In the shell code, please change the hyper-parameters referred to [Hyperparameter Setting](#hyperparameter). For example, ![](https://latex.codecogs.com/svg.image?&space;\lambda_{u}) is equal to *lamb_u*, ![](https://latex.codecogs.com/svg.image?&space;\lambda_{i}) is equal to *lamb_i*, abd ![](https://latex.codecogs.com/svg.image?&space;e_{max}) is equal to *e_max*. 

#### MELT+SASRec  

``` python  
# In the shell code, please change the 'data' variable 
bash shell/MELT_SASRec/train_amazon.sh # Dataset: Clothing, Sports, Beauty, Grocery, Automotive, Music 
```  

#### MELT+FMLP  

``` python  
# In the shell code, please change the 'data' variable 
bash shell/MELT_FMLP/train_amazon.sh # Dataset: Clothing, Sports, Beauty, Grocery, Automotive, Music 
```  

Additionally, we provide the trained MELT model for each dataset. Following the below link, you can download them.

* [MELT-SASRec](https://drive.google.com/drive/folders/1RjlsxNGat1eMICVZ9sLE3Q1pb2Xxjp5n?usp=sharing)  

* [MELT-FMLP](https://drive.google.com/drive/folders/1BoPHgc-c1MPZUHWu8lJ3ldclNAEA4OVy?usp=sharing)  

When you download the MELT's pre-trained model, put it on *save_model/{DATA_NAME}* directory.  


To train the model on **Behance** or **Foursquare** datasets, please run the **train_others.sh** shell code.


## **Inference**


### SASRec  
``` python  
# In the shell code, please change the 'data' variable 
bash shell/SASRec/test_amazon.sh # Dataset: Clothing, Sports, Beauty, Grocery, Automotive, Music 
```  


### MELT-SASRec  
``` python  
# In the shell code, please change the 'data' variable 
bash shell/MELT_SASRec/test_amazon.sh # Dataset: Clothing, Sports, Beauty, Grocery, Automotive, Music 
```  
To test the model on **Behance** or **Foursquare** datasets, please run the **test_others.sh** shell code.


## **Algorithm**  

To better understand the MELT framework, we provide the [algorithm](algorithm.md). 

## **Hyperparameter**  

* **MELT+SASRec**


|Data|lamb_U|lamb_I|e_max|Pareto(%) - a|  
|---|---|---|---|---|  
|Music|0.2|0.3|180|20|  
|Automotive|0.1|0.4|180|20|  
|Beauty|0.1|0.4|180|20|  
|Sports|0.4|0.3|200|20|  
|Clothing|0.3|0.2|200|20|  
|Grocery|0.1|0.3|180|20|  
|Foursquare|0.1|0.1|200|50|  
|Behance|0.1|0.2|180|50|

* **MELT+FMLP**  

|Data|Lamb_U|Lamb_I|E_max|Pareto(%) - a|  
|---|---|---|---|---|  
|Music|0.1|0.1|200|20|  
|Automotive|0.3|0.3|160|20|  
|Beauty|0.2|0.1|200|20|  
|Sports|0.2|0.3|180|20|  
|Clothing|0.4|0.3|180|20|  
|Grocery|0.2|0.2|200|20|  
|Foursquare|0.1|0.1|200|50|   
|Behance|0.1|0.3|200|50|


## **Citation**  
```  
@inproceedings{kim2023melt,
  title={MELT: Mutual Enhancement of Long-Tailed User and Item for Sequential Recommendation},
  author={Kim, Kibum and Hyun, Dongmin and Yun, Sukwon and Park, Chanyoung},
  booktitle={Proceedings of the 46th international ACM SIGIR conference on Research and development in information retrieval},
  pages={68--77},
  year={2023}
}
```


