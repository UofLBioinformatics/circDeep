# circDeep: End-to-End learning framework for circular RNA classification from other long non-coding RNA using multimodal deep learning
circDeep fuse RCM descriptor, ACNN-BLSTM sequence descriptor and conservation descriptor into high level abstraction descriptors, where the shared representations across different modalities are integrated. The experiments show that circDeep is not only faster than existing tools but also performs at an unprecedented level of accuracy by achieving a 12 percent increase in accuracy over other existing tools.

Authors: [Bioinformatics Lab](http://bioinformatics.louisville.edu/lab/index.php), [Kentucky Biomedical Research Infrastructure Network (KBRIN)](http://louisville.edu/research/kbrin/), University of Louisville

## Prerequisites
We recommend to use [Anaconda 3](https://www.anaconda.com/download/) platform. 
- [Keras](https://anaconda.org/conda-forge/keras) (Deep learning library)

- [scikit-learn](https://anaconda.org/anaconda/scikit-learn) (Machine learning library)

- [h5py](https://anaconda.org/anaconda/h5py)

- [gensim](https://anaconda.org/anaconda/h5py) 

- [pysam](https://anaconda.org/bioconda/pysam) >= 0.9.1.4

- [pybigwig](https://anaconda.org/bioconda/pybigwig)

## Installation
Download circDeep by
```bash 
git clone https://github.com/UofLBioinformatics/circDeep
```
Installation has been tested in Anaconda (Linux/Windows) platform with Python3.

## Usage

```bash 
usage: circDeep.py [-h] --train TRAIN --genome GENOME -gtf GTF --bigwig BIGWIG
               [--seq SEQ] [--rcm RCM] [--cons CONS] [--predict PREDICT]
               [--out_file OUT_FILE] [--model_dir MODEL_DIR] 
               [--positive_bed POSITIVE_BED] [--negative_bed NEGATIVE_BED] 
               [--testing_bed TESTING_BED] 

circular RNA classification from other long non-coding RNA using multimodal deep learning

Required arguments:
=================== 
   --data_dir <data_directory>
                        Under this directory, you will have descriptors files used for training, the label file, genome sequencefile , gtf annotation file and bigwig file
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
## Example
#### Train the model:
In our experiements, we have used [circular RNAs](https://raw.githubusercontent.com/UofLBioinformatics/circDeep/master/data/circRNA_dataset.bed) from [circRNADb](http://202.195.183.4:8000/circrnadb/circRNADb.php) and our [negative dataset](https://raw.githubusercontent.com/UofLBioinformatics/circDeep/master/data/negative_dataset.bed) from [GENCODE](https://www.gencodegenes.org/). The original coordinates of our datasets were in hg19 genome and we convert them to hg38 genome using [liftOver](https://genome.ucsc.edu/cgi-bin/hgLiftOver) provided in [UCSC Genome Browser](https://genome.ucsc.edu/). We need also to download all necessary files and put them in data directory.
- Dowload genome sequence in FASTA format for human genome ( It can be downloaded from [UCSC Genome Browser](https://genome.ucsc.edu/))
- Dowload [gtf annotation for human genome](https://useast.ensembl.org/info/data/ftp/index.html).
- Download [phastCons scores](http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons20way/) for the human genome in PhastCons format.  
```bash
python3 circDeep.py --data_dir 'data/' --train True --model_dir 'models/' --seq True --rcm True --cons True --genome 'data/hg38.fasta' --gtf 'data/Homo_sapiens.Ensembl.GRCh38.82.gtf' --bigwig 'data/hg38.phastCons20way.bw' --positive_bed 'data/circRNA_dataset.bed' --negative_bed 'data/negative_dataset.bed'
```
#### Test the model:
```bash
python3 circDeep.py --data_dir 'data/' --train False --model_dir 'models/' --seq True --rcm True --cons True --genome 'data/hg38.fasta' --gtf 'data/Homo_sapiens.Ensembl.GRCh38.82.gtf' --bigwig 'data/hg38.phastCons20way.bw' --testing_bed 'data/test.bed'
```
## License

Copyright (C) 2017 .  See the [LICENSE](https://github.com/UofLBioinformatics/circDeep/blob/master/License)
file for license rights and limitations (MIT).
