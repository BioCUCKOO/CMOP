#! /usr/bin/perl -w
use strict;
use Statistics::Distributions qw(chisqrprob);

###
if(@ARGV != 1){
	print "Error: require only one .ini file, please run such as: perl ~.pl pMaT.ini. \n\n";
	exit;
}

### .ini file
open INI,"$ARGV[0]" or die $!;
my $sample="";
my $relation="";
my $matrix="";
my $cutoff="";
my $Output="";
while(<INI>){
	chomp;
	my @a=split(/\t/,$_);
	if($_=~/^#/){
		next;
	}elsif($a[0] eq "Sample information"){
		$sample=$a[1];
	}elsif($a[0] eq "TF-target relations"){
		$relation=$a[1];
	}elsif($a[0] eq "Transcriptomic matrix"){
		$matrix=$a[1];
	}elsif($a[0] eq "Cutoff"){
		$cutoff=$a[1];
	}elsif($a[0] eq "Output folder"){
		$Output=$a[1];
	}
}

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

### TF-target relations
my %relation;
if(-e $relation){
	open REL,"$relation" or die $!;
	while(<REL>){
		chomp;
		my @a=split(/\t/,$_);
		
	}
}else{
	last;
	print "Error: please set the path to file of TF-target relations.\n";
}

###


