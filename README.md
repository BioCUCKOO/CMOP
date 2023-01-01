# ALS and pAKin
The two algorithms were developed for multi-omic integration and analysis of ICC, including an artificial intelligence (AI) framework, autoencoder + LASSO for subtyping (ALS) for an integrated molecular subtyping and prioritization of actionable kinases (pAKin) for prediction of potentially functional ICC-associated kinases.
<br>

# The description of each source code
### ALS.py
The code of ALS framework for the integrated molecular subtyping. 
<br>

### pAKin.pl
The program infers actionable kinases from the integration of transcriptomic, proteomic and phosphoproteomic features. The usage of the code is shown as below: <br><br>
perl pAKin.pl conf.ini
<br>

### Example_data
This folder contains example data and files for testing. It should be noted that some files are partially present due to the limitation of uploading size, such as "Expreesion.matrix".

# Computation Requirements
### OS Requirements
Above codes have been tested on the following systems: <br>
Windows: Windows 7, Windos 10<br>
Linux: CentOS linux 7.8.2003

### Software Requirements
Python (version 3.6 or later) tool with packages (Keras, tensorflow and scikit-learn), R (version 4.0.3 or later) tool with packages (survival and survminer), Perl (v5.26.3 or later) program with modules (Statistics::Distributions).

### Hardware Requirements
All codes and softwares could run on a "normal" desktop computer, no non-standard hardware is needed.<br>
<br>

# Contact
Wanshan Ning: ningwanshan@hust.edu.cn
Shaofeng Lin: linshaofeng@hust.edu.cn

