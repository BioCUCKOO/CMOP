#! /usr/bin/perl -w
use strict;
use Statistics::Distributions qw(chisqrprob);

###
if(@ARGV != 1){
	print "Error: require only one .ini file, please run such as: perl ~.pl pAKin.ini. \n\n";
	exit;
}

### .ini file
open INI,"$ARGV[0]" or die $!;
my $sample="";
my $ssksr="";
my $transcriptomic="";
my $Tcutoff="";
my $proteomic="";
my $Pcutoff="";
my $Phosphoproteomic="";
my $Netcutoff="";
my $Intcutoff="";
while(<INI>){
	chomp;
	my @a=split(/\t/,$_);
	if($_=~/^#/){
		next;
	}elsif($a[0] eq "Sample information"){
		$sample=$a[1];
	}elsif($a[0] eq "Site-specific kinase-substrate relations"){
		$ssksr=$a[1];
	}elsif($a[0] eq "Transcriptomic matrix"){
		$transcriptomic=$a[1];
	}elsif($a[0] eq "Transcriptomic cutoff"){
		$Tcutoff=$a[1];
	}elsif($a[0] eq "Proteomic matrix"){
		$proteomic=$a[1];
	}elsif($a[0] eq "Proteomic FC-cutoff"){
		$Pcutoff=$a[1];
	}elsif($a[0] eq "Phosphoproteomic matrix"){
		$Phosphoproteomic=$a[1];
	}elsif($a[0] eq "Network-cutoff"){
		$Netcutoff=$a[1];
	}elsif($a[0] eq "Intensity-cutoff"){
		$Intcutoff=$a[1];
	}
}
close INI;


### Sample
my %sample;
if (-e $sample){
	open SAM,"$sample" or die $!;
	while(<SAM>){
		chomp;
		my @a=split(/\t/,$_);
		if($a[0] eq "Sample"){
			next;
		}else{
			$sample{$a[2]}=$a[1];
		}
	}
	close SAM;
}else{
	last;
	print "Error: please set the path to file of sample information.\n";
}


### 




