# RDS, pAKin and pMaT
The three novel algorithms were developed for multi-omic integration and analysis in ICC, including random dropout-based subtyping (RDS) for an improved molecular subtyping, prioritization of actionable kinases (pAKin) for prioritization of potentially actionable kinases, and prioritization of master TFs (pMaT) for prediction of key TFs.
<br>

# The description of each source code
### RDS.R
The code for the original and improved molecular subtyping with corresponding clinical outcome, such as the overall survival.
<br>

### pAKin.pl
The program infers actionable kinases from the integration of transcriptomic, proteomic and phosphoproteomic features.
<br>

### pMaT.pl
Predicts potentially master TFs based on the transcriptomic data.
<br>

### Example_data
This folder contains example data and files for testin. It should be noted that some files are partially present due to the limitation of uploading size, such as "Expreesion.matrix".

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
