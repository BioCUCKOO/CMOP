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

###
open SUM,"2--TFs-Sum-FPKM-Sqrt.txt" or die $!;
my %sum;
while(<SUM>){
	chomp;
	my @a=split(/\t/,$_);
	$sum{$a[0]}=$a[1];
}
close SUM;

###
open TF,"2--TFs-Cancer-FPKM-Sqrt.txt" or die $!;
open OUT,"> 3--TFs-vs-FPKM-Sqrt-p.txt" or die $!;
my %vs;	# Sample; Ca; P
while(<TF>){
	chomp;
	my @a=split(/\t/,$_);
	if($a[0] eq "ID"){
		foreach my $k(2..$#a){
			my $tmp=$a[$k];
			$tmp=~s/Ca_//;
			$tmp=~s/P_//;
			if(exists $vs{$tmp}){
				$vs{$tmp}=$vs{$tmp}."; ".$k;
			}else{
				$vs{$tmp}="$tmp; $k";
			}
		}
		##
		print OUT "ID\tName";
		foreach my $k(sort keys %vs){
			print OUT "\t$k,p";
		}
		print OUT "\n";
	}else{
		print OUT "$a[0]\t$a[1]";
		foreach my $k(sort keys %vs){
			my @b=split(/; /,$vs{$k});
			my $n=$a[$b[1]];
			my $sumn="P_$b[0]";
			$sumn=$sum{$sumn};
			my $t=$a[$b[2]];
			my $sumt="Ca_$b[0]";
			$sumt=$sum{$sumt};
			my $e="NA";
			my $s="NA";
			my $p;
			if($n eq "NA" | $t eq "NA"){
				$p="NA";
			}else{
				$e=$t/$sumt/($n/$sumn);
				$s=&yates($n, $t, ($sumn-$n), ($sumt-$t));
				$p=&chisqrprob(1,$s)+0;
			}
			# print OUT "\t$e";
			# print OUT "\t$s";
			print OUT "\t$p";
		}
		print OUT "\n";
	}
}
close TF;












### Sub
sub yates{
	my ($a,$b,$c,$d)=@_;
	my $n=$a+$b+$c+$d;
	#print "A:$a\t$b\t$c\t$d\t$n\n";
	my $s=0;
	if(($a+$b)*($a+$c)/$n<5 or ($a+$b)*($b+$d)/$n<5 or ($c+$d)*($a+$c)/$n<5 or($c+$d)*($b+$d)/$n<5 ){
		$s=$n*((abs($a*$d-$b*$c))-$n/2)**2/(($a+$b)*($c+$d)*($a+$c)*($b+$d));
	}else{
		$s=$n*($a*$d-$b*$c)**2/(($a+$b)*($c+$d)*($a+$c)*($b+$d));
	}
	#print "$s\n";
	return $s;
}




