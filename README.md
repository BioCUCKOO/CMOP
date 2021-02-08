# RDS, pAKin and pMaT
The three novel algorithms were developed for multi-omic integration and analysis of ICC, including random dropout-based subtyping (RDS) for an improved molecular subtyping, prioritization of actionable kinases (pAKin) for prediction of potentially functional ICC-associated kinases, and prioritization of master TFs (pMaT) for prediction of potentially master TFs in ICC.
<br>

# The description of each source code
### RDS.R
The code for the original and improved molecular subtyping with corresponding clinical outcome, such as the overall survival. The usage of the code is shown in code file.
<br>

### pAKin.pl
The program infers actionable kinases from the integration of transcriptomic, proteomic and phosphoproteomic features. The usage of the code is shown as below: <br>
perl pAKin.pl conf.ini
<br>

### pMaT.pl
Predicts potentially master TFs based on the transcriptomic data. The usage of the code is shown as below: <br>
perl pMaT.pl conf.ini
<br>

### Example_data
This folder contains example data and files for testing. It should be noted that some files are partially present due to the limitation of uploading size, such as "Expreesion.matrix".

# Computation Requirements
### OS Requirements
Above codes have been tested on the following systems: <br>
Windows: Windows 7, Windos 10<br>
Linux: CentOS linux 7.8.2003

### Software Requirements
R (version 4.0.3 or later) tool with packages (ConsensusClusterPlus, survival and survminer), Perl (v5.26.3 or later) program with modules (Statistics::Distributions).

### Hardware Requirements
All codes and softwares could run on a "normal" desktop computer, no non-standard hardware is needed.<br>
<br>

# Contact
Dr. Yu Xue: xueyu@hust.edu.cn<br>
Shaofeng Lin: linshaofeng@hust.edu.cn

