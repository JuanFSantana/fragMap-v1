# fragMap.py #
Juan F. Santana, Ph.D. (<juan-santana@uiowa.edu>), University of Iowa, Iowa City, I.A.

This improved fragMap program is simplified, more powerful, and runs faster than the original fragMap program (https://github.com/P-TEFb/fragMap/blob/main/original_fragMap_programs.md). It can now also take replicas and their corrections factors creating fragMaps for each of the replicas. It will also create a fragMap from combining the spike-in corrected replicas. The program requires a Linux operating system and Python 3+. It will create fragment heatmaps from a specific range of fragment sizes over a chosen genomic interval as described in [Santana et al., 2022](https://academic.oup.com/nar/article/50/16/9127/6659871) and [Spector et al., 2022](https://www.nature.com/articles/s41467-022-29739-x). This program replaces the original fragMap program created by Mrutyunjaya Parida that was used to generate fragMaps in [Spector et al., 2022](https://www.nature.com/articles/s41467-022-29739-x), [Ball et al., 2022a](https://www.mdpi.com/1999-4915/14/4/779), [Ball et al., 2022b](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9239164/), and [Santana et al., 2022](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkac678/6659871?guestAccessKey=88024805-7d8e-4421-a032-dbef1c737757).

# File requirements #
The input regions file should be a six-column, tab-delimited bed file containing chromosome, start and end positions, and the strand information for each region. The regions can be of any length as long as it is an even number and the center is a feature under study (e.g. transcription start site). 
 
| chr6 | 142946246 | 142946446 | Gene_A | 255 | - |
|:----:|:---------:|:---------:|:------:|:---:|:-:|

The input fragments file should be a six-column, tab-delimited bed file containing chromosome, start and end positions, and the strand information for each fragment.

| chr6 | 142946247 | 142946298 | A00876:119:HW5F5DRXX:2:2207:29170:1157 | 255 | - |
|:----:|:---------:|:---------:|:--------------------------------------:|:---:|:-:|


# Behavior #
Generates a fragMap from a specific range of fragment sizes over a chosen genomic interval. 

# Dependencies #
### Python libraries ###
Pandas: https://pypi.org/project/pandas/

Numpy: https://pypi.org/project/numpy/

Matplotlib: https://matplotlib.org/stable/users/installing/index.html

Pillow: https://pillow.readthedocs.io/en/stable/index.html

bedtools: https://bedtools.readthedocs.io/en/latest/content/installation.html, developed by the Quinlan laboratory at the University of Utah. 

# Example command usage #
```
python3 fragMap.py plusminus1000_from_TSS_1000genes.bed \
                  -f PolII-DFF-ChIP-Seq-Rep1.bed PolII-DFF-ChIP-Seq-Rep2.bed PolII-DFF-ChIP-Seq-Rep3.bed \
                  -o /home/user/dir/ \
                  -r 20 400 \
                  -y 4 \
                  -g 0.5 \
		  -n Rep1 Rep2 Rep3 \
                  -s 1 1.3 1.5

```
# Parameter description #
```
regions: <str> Bed file of genomic regions of chosen length with the format described above

-f: <str> Bed file(s) of fragment positions with the format described above. Multiple data files (replicas) can be added at the same time

-r: <int> <int> Range of fragment sizes, for example -r 20 400

-s: <int | float> Correction factors for each data file in `-f` in the same order

-b: <int> Sets the chosen value as black, default is the largest number in the matrix

-c: If invoked, the output will be a fragMap of centers of fragments

-y: <int> (value greater than or equal to 1) Horizontal lines/bp for each fragment length, default is 1

-x: <float> or <int> (value less than or equal to 1) Vertical lines/bp for each genomic interval displayed, for example, -x 1 is one vertical line/bp; -x 0.1 is one vertical line/averaged 10 bp, default is 1

-g: <float> Gamma correction factor, default is 1 but 0.5 provides an image which is more interpretable by the human eye. For more information: https://en.wikipedia.org/wiki/Gamma_correction

-o: <str> Ouput directory

-n: <str> Image identifier(s) for each data file in `-f` in the same order

```
Example output from Pol II DFF-Seq performed on HFF cells ([Spector et al., 2022](https://www.nature.com/articles/s41467-022-29739-x)) over +/- 1,000 bp regions from the MaxTSS of 12,229 genes in HFF cells determined with PRO-Cap ([Nilson et al., 2017](https://academic.oup.com/nar/article/45/19/11088/4084663)): 

PolII_fragMap_20-400_Max_68609_X_1_Y_4_**Gamma_1.0** 
![PolII_fragMap_Custom_20-400_Max_68609_X_1 0_Y_4 0_](https://user-images.githubusercontent.com/38702786/190675335-1b8271ef-a0f7-449e-9ac3-aeee7dca6611.png)

PolII_fragMap_20-400_Max_68609_X_1_Y_4_**Gamma_0.5**
![PolII_fragMap_20-400_Max_68609_X_1 0_Y_4 0_Gamma_0 5_](https://user-images.githubusercontent.com/38702786/191344898-fc082eb6-6c3c-4b12-a6f1-8ef62ef5047c.png)
