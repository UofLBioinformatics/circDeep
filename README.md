# circDeep: End-to-End learning framework for   circular RNA classification from other long non-coding RNA using multimodal deep learning
circDeep fuse RCM descriptor, ACNN-BLSTM sequence descriptor and conservation descriptor into high level abstraction descriptors, where the shared representations across different modalities are integrated. The experiments show that circDeep is not only faster compared to existing tools but also performs unprecedented level of accuracy by achieving a gap of over 12% increase.

Authors: [Bioinformatics Lab](http://bioinformatics.louisville.edu/lab/index.php), University of Louisville, [Kentucky Biomedical Research Infrastructure Network (KBRIN)](http://louisville.edu/research/kbrin/)

## Prerequisites

- [Keras](https://github.com/antoniosehk/keras-tensorflow-windows-installation)

- [sklearn](https://github.com/scikit-learn/scikit-learn) 

- [h5py](https://anaconda.org/conda-forge/h5py)

- [gensim](https://anaconda.org/anaconda/gensim) 

- [pysam](https://github.com/pysam-developers/pysam) >=0.9.1.4

-[pybigwig](https://bioconda.github.io/recipes/pybigwig/README.html)


## Installation
1 Download circDeep
```bash
git clone https://github.com/UofLBioinformatics/circDeep.git

```

2 Install required packages

3 Install circDeep
```bash
python setup.py install
```
4 testing circDeep with testrun.sh

In order to run testrun.sh:

- Dowload genome sequence in FASTA format for Rattus_norvegicus genome ( It can be downloaded from [UCSC Genome Browser](https://genome.ucsc.edu/)) 
- Dowload [gtf annotation for Rattus_norvegicus genome](https://www.ncbi.nlm.nih.gov/).
- Run testrun.sh to test the installation and the dependencies of seekCRIT and also it will test the installation with  [CRTL12.fastq](https://github.com/UofLBioinformatics/seekCRIT/blob/master/seekCRIT/testData/CTRL.fastq) and [IR12.fastq](https://github.com/UofLBioinformatics/seekCRIT/blob/master/seekCRIT/testData/IR12.fastq) by specifying the path for FASTA and gtf files:
```bash
 ./testrun.sh gtf/Rattus_norvegicus.Ensembl.rn6.r84.gtf fasta/rn6.fa
```

## Usage

```bash 
usage: circDeep.py [-h] -s1 S1 -s2 S2 -gtf GTF -o OUTDIR -t {SE,PE} --aligner
               ALIGNER --genomeIndex GENOMEINDEX -fa FASTA -ref REFSEQ
               [--threadNumber numThreads]
               [--deltaPSI DELTAPSI] [--highConfidence HIGHCONFIDENCE]
               [--libType {fr-unstranded,fr-firststrand,fr-secondstrand}]
               [--keepTemp {Y,N}]

Identifying and Characterizing Differentially Spliced circular RNAs between
two samples

Required arguments:
=================== 
   --data_dir <data_directory>
                        Under this directory, you will have descriptors files
  --train TRAIN         use this option for training model
  --genome GENOME       Genome sequence. e.g., hg38.fa
  --gtf GTF             The gtf annotation file. e.g., hg38.gtf
  --bigwig BIGWIG       conservation scores in bigWig file format
                        
 optional arguments:
====================

   -h, --help            show this help message and exit
  --seq SEQ             The modularity of ACNN-BLSTM seq
  --rcm RCM             The modularity of RCM
  --cons CONS           The modularity of conservation
  --predict PREDICT     Predicting circular RNAs. if using train, then it will
                        be False
  --out_file OUT_FILE   The output file used to store the prediction
                        probability of testing data
  --model_dir MODEL_DIR
                        The directory to save the trained models for future
                        prediction
   --positive_bed POSITIVE_BED
                        BED input file for circular RNAs for training, it
                        should be like:chromosome start end gene
  --negative_bed NEGATIVE_BED
                        BED input file for other long non coding RNAs for
                        training, it should be like:chromosome start end gene
  --testing_bed TESTING_BED
                        BED input file for testing data, it should be
                        like:chromosome start end gene
```
### Example
#### Paired-end reads
```bash
python3 seekCRIT.py -o PEtest -t PE --aligner STAR -fa fa/hg19.fa -ref ref/hg19.ref.txt --genomeIndex /media/bio/data/STARIndex/hg19 -s1 testData/231ESRP.25K.rep-1.R1.fastq:testData/231ESRP.25K.rep-1.R2.fastq,testData/231ESRP.25K.rep-2.R1.fastq:testData/231ESRP.25K.rep-2.R2.fastq -s2 testData/231EV.25K.rep-1.R1.fastq:testData/231EV.25K.rep-1.R2.fastq,testData/231EV.25K.rep-2.R1.fastq:testData/231EV.25K.rep-2.R2.fastq -gtf testData/test.gtf --threadNumber 12 
```
#### Single-end reads

```bash
python3 seekCRIT.py -o SEtest -t SE --aligner STAR -fa fa/hg19.fa -ref ref/hg19.ref.txt --genomeIndex /media/bio/data/STARIndex/hg19 -s1 testData/231ESRP.25K.rep-1.R1.fastq,testData/231ESRP.25K.rep-1.R2.fastq,testData/231ESRP.25K.rep-2.R1.fastq,testData/231ESRP.25K.rep-2.R2.fastq -s2 testData/231EV.25K.rep-1.R1.fastq,testData/231EV.25K.rep-1.R2.fastq,testData/231EV.25K.rep-2.R1.fastq,testData/231EV.25K.rep-2.R2.fastq -gtf testData/test.gtf --threadNumber 12 
```





## Note
- Transcriptome should be in refseq format below (see more details in the [example](https://github.com/UofLBioinformatics/seekCRIT/blob/master/example/hg19.ref.txt) ):

| Field                           | Description                                                                  |
| :------------------------------:|:---------------------------------------------------------------------------- |
| geneName                        | Name of gene                                                                 |
| isoform_name                    | name of isoform                                                              |
| chrom                           | chromosme                                                                    |
| strand                          |  strand  (+/-)                                                             |
| txStart                         | Transcription start position                                                 |
| txEnd                           | Transcription end position                                                   |
| cdsStart                        |Coding region end   		                                                       |
| exonCount						            | 		    Number of exons 		                    						                 |
| exonStarts					            | 		 Exon start positions       								                             |
| exonEnds						            | 		    Exon end positions           						                             |

- It is not obligatory to provide REFSEQ file, we made script (GTFtoREFSEQ) to convert from gtf to refseq that is used in the main code if no refseq file is provided.


## Output

See details in [the example file](https://github.com/UofLBioinformatics/seekCRIT/blob/master/example/circRNAs.pVal.FDR.txt)

| Field                           | Description                                                                  |
| :------------------------------:|:---------------------------------------------------------------------------- |
| chrom                           | chromosome                                                                   |
| circRNA_start                   | circular RNA 5' end position                                                 |
| circRNA_end                     | circular RNA 3' end position                                                 |
| strand                          | DNA strand  (+/-)                                                            |
| exonCount                       | number of exons included in the circular RNA transcript                      |
| exonSizes                       | size of exons included in the circular RNA transcript                        |
| exonOffsets                     | offsets of exons included in the circular RNA transcript  					         |  
| circType					          	  | 		    circRNA, ciRNA, ccRNA    						                    		                             |
| geneName						            | 		    name of gene    		                    						                             |
| isoformName					            | 		  name of isoform                           								                             |
| exonIndexOrIntronIndex		      |         Index (start from 1) of exon (for circRNA) or intron (for ciRNA) in given isoform            		        								                             |
| FlankingIntrons			        	  | 		        Left intron/Right intron                     								                             |
| CircularJunctionCount_Sample_1  | read count of the circular junction in sample # 1                            |
| LinearJunctionCount_Sample_1	  | 	       read count of the linear junction in sample # 1                 											                             |
| CircularJunctionCount_Sample_2  | 	      read count of the circular junction in sample # 2            											                             |
| LinearJunctionCount_Sample_2	  | 		  read count of the linear junction in sample # 1                       										                             |
| PBI_Sample_1					          | 		  Percent Backsplicing Index for  sample # 1                       										                             |
| PBI_Sample_2				         	  | 				 Percent Backsplicing Index for  sample # 2                         								                             |
| deltaPBI(PBI_1-PBI_2)			      | 			 difference between PBI values of two samples                           							                             |
| pValue						              | 				   pValue                      								                             |
| FDR							                |               FDR           												                             |


## License

Copyright (C) 2017 .  See the [LICENSE](https://github.com/UofLBioinformatics/seekCRIT/blob/master/LICENSE)
file for license rights and limitations (MIT).
