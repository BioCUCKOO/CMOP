library("readr")
library("stringi")
library("xlsx")
library("tidyverse")
library("survival")
library("survminer")
library("dplyr")

datapath="sur.txt"
ori=read.csv(datapath,sep="\t",header =TRUE,fileEncoding = 'utf-8',colClasses = c('numeric',"character",rep('numeric',32)))
tdata=ori[,-c(1,2)]
fit <- survfit(Surv(days,status)~int,data=tdata)
ggsurvplot(fit,pval = TRUE, conf.int = TRUE,ggtheme = theme_bw())
ggsurvplot(fit,risk.table=TRUE,
           
           conf.int=FALSE,
           
           palette = c("#32cd32", "#2E9FDF","#e9565f","black"),
           
           pval=TRUE,
           
           pval.method=TRUE)
		   
res.cox <- coxph(Surv(days,status) ~ int, data = tdata)
res.cox
summary(res.cox)
