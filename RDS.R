### set work directory
setwd("Your directory")

### Installation of packages


### Library
library(ConsensusClusterPlus)
library(survival)
library(survminer)

### Dataset
## Omicsdata
rawdata <- read.table(file='Example_data/RDS.matrix',header=T,row.names=1,sep="\t",check.names=F,stringsAsFactors=FALSE )
dim(rawdata)
new <- as.matrix(samrdata)
## Survival data
Sur <- read.table(file='Example_data/OS.txt',header=T,sep="\t",check.names=F,stringsAsFactors=FALSE )
dim(Sur)

### Original molecular subtyping
## Output file
pfile="Clustering-Original-p.txt"
unlink(pfile)
header=paste("Cluster", "Distance", "Seed", "Folder", "p_value","1","2","3", sep="\t")
write(header,file=pfile, append=TRUE)
## Cluster method
{
## km
cluster="km"
dlist=c("euclidean")
## hc
cluster="hc"
dlist=c('pearson', 'spearman' , 'euclidean', 'binary', 'maximum', 'canberra', 'minkowski')
## pam
cluster="pam"
dlist=c('pearson', 'spearman' , 'euclidean', 'binary', 'maximum', 'canberra', 'minkowski')
}
## 
seedlist=c(1,9,99,999,2020,9999,99999)
for (i in dlist){
	distance=i
	for (l in seedlist){
		seed=l
		folder=paste(cluster,distance,seed,sep="-")
		results = ConsensusClusterPlus(new,maxK=6,reps=1000,pItem=0.8,pFeature=0.8, title=folder,clusterAlg=cluster,distance=distance,seed=seed,plot="png")
		classfile=paste(folder,"/3Class.txt",sep="")
		write.table(results[[3]][["consensusClass"]],file=classfile,sep="\t",quote=F, col.names = NA)
		### 
		Months <- data.frame(Sample=Sur$Label,Day=Sur$Days,Month=Sur$Months,Status=Sur$Status, type=results[[3]][["consensusClass"]][Sur$Label])
		num <- as.numeric(table(Months$type))
		num <- paste(num, collapse ="\t")
		##
		fit <- survfit(Surv(Month,Status)~type, data = Months)
		surv_diff <- survdiff(Surv(Month,Status) ~ type, data = Months)
		surv_diff
		p_value <- 1 - pchisq(surv_diff$chisq, length(surv_diff$n) -1)
		p_value
		##
		out=paste(cluster, distance, seed, folder, p_value, num, sep="\t")
		print(c(cluster, distance, seed, folder, p_value, num))
		write(out,file=pfile, append=TRUE)
	}
}



### Improved molecular subtyping
## Droupout Function
DroupoutFun <- function(num, seednew, seed, inputdata, droupout, plist, last, plast, cluster, distance, Sur){
	## File
	out=paste("Droupout", num, seednew, seed, sep="-");
	outfile=paste(out,"-p.txt",sep="")
	header=paste("Cluster", "Distance", "Seed", "Seednew", "Index", "Left", "p_value", "1","2","3", "Last", sep="\t")
	write(header, file=outfile)
	## 
	maxnum = nrow(inputdata)
	run = maxnum%/%num
	print(c(num,out,outfile,maxnum,run))
	dir.create(out)
	## Droupout
	now=2
	for(i in 2:run){
		set.seed(num)
		tmp=-sample(1:maxnum, num*i, replace = FALSE)
		tmp=tmp[(num*i-num+1):(num*i)]
		droup=c(last,tmp)
		new <- inputdata
		nrow(new)
		new <- new[droup,]
		nrow(new)
		##
		folder=paste(out, "/" ,i,sep="")
		new <- as.matrix(new)
		results = ConsensusClusterPlus(new,maxK=6,reps=1000,pItem=0.8,pFeature=0.8, title=folder,clusterAlg=cluster,distance=distance,seed=seed,plot="png")
		## 
		classfile=paste(folder,"/3Class.txt",sep="")
		write.table(results[[3]][["consensusClass"]],file=classfile,sep="\t",quote=F, col.names = NA)
		##
		idfile=paste(folder,"/id.txt",sep="")
		write.table(rownames(new),file=idfile,sep="\t",quote=F, col.names = NA)
		##
		Months <- data.frame(Sample=Sur$Label,Day=Sur$Days,Month=Sur$Months,Status=Sur$Status, type=results[[3]][["consensusClass"]][Sur$Label])
		classnum <- as.numeric(table(Months$type))
		classnum <- paste(classnum, collapse ="\t")
		##
		fit <- survfit(Surv(Month,Status)~type, data = Months)
		surv_diff <- survdiff(Surv(Month,Status) ~ type, data = Months)
		# surv_diff
		p_value <- 1 - pchisq(surv_diff$chisq, length(surv_diff$n) -1)
		p_value
		if(p_value<=plast){
			plist[[i]]=p_value
			droupout[[i]]=droup
			plast=p_value
			last=droup
			now=i
		}else{
			plist[[i]]=1
			droupout[[i]]=last
		}
		pout=paste(cluster, distance, seed, seednew, i, nrow(new), p_value, classnum, now, sep="\t")
		write(pout,file=outfile, append=TRUE)
		print(c(i,"OS-p:", p_value, classnum, plast, now))
		if(i-now>20){
			break
		}
	}
	print (c("FC:",plist[[1]]/plast))
	write(plist[[1]]/plast,file=outfile, append=TRUE)
	return (plist[[1]]/plast)
}
## Initialization
droupout=list()
droupout[[1]]=c(0)
plist=list()
## Cluster method and p-value obtained from original molecular subtyping
cluster="pam"
distance='spearman'
plist[[1]]=0.00450366545163683
## Calls Droupout Function
last=droupout[[1]]
plast=plist[[1]]
for(i in 10:1){
	print(i)
	DroupoutFun(i, 999, 999, samrdata, droupout, plist,last, plast, cluster, distance, Sur)
}

