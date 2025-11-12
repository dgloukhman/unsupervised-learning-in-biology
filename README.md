# Unsupervised Learning in Biology

This repository aims to replicate the findings presented in the paper [Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.](https://pubmed.ncbi.nlm.nih.gov/33876751/) The project focuses on exploring and implementing unsupervised learning techniques for biological sequence analysis, specifically protein sequences, to understand how structural and functional insights can be derived from large-scale unlabeled data.


## Download Data

To download the [UniParc]() dataset execute:

```
wget -P data -r -nd --no-parent -A 'uniparc_active_p*.fasta.gz' https://ftp.expasy.org/databases/uniprot/current_release/uniparc/fasta/active -e robots=off
```