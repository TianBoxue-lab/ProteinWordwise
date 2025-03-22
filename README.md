# ProteinWordwise


`ProteinWordwise` is a protein functional analysis tool based on ESM2 pretrained protein language model. 

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Usage](#usage)
- [License](#license)

---

# Overview

``Protein Wordwise``  is a protein functional analysis tool, aim to parse analyte protein sequences into function-related residual clusters, or "protein words".By analyzing the attention matrices of ESM2 through Louvain community detection algorithmWord2Function, the protein sequences are parsed. The raw words can be furtherly filtered using automatically pre-defined "dictionaries" to priortize functionally informative protein words. A downstream sequence-function prediction model based on the protein words defined in ProteinWordwise and its pretrained weights, is also provided in this toolkit.

# System Requirements

## Hardware requirements

To run ProteinWordwise on single protein with ready-made general and family-specific dictionaries, a computer with the capability to run inference of its fundamental model, [ESM2](https://github.com/facebookresearch/esm) (T33-650B version) is required.
Creating dictionary from your own protein sequence dataset is an memory-consuming task, depending on your dataset size. 16GB RAM can run this step for about 1000 sequences with an average sequence length of 500.

## Software Dependencies

`ProteinWordwise` is a Python-based tool that requires an environment to run ESM2. The workflow has been tested and evaluated in the following Python enviroment:

```
python=3.10.9
numpy=1.23.5
pandas=1.5.2
torch=2.3.1
esm=2.0.0
networkx=3.2.1
scipy=1.15.1
biopython=1.83
```

# Usage

## Extracting Word List from Sequence

To run ProteinWordwise word list extraction with your sequences using our ready-made Pfam family and UniRef50 general dictionaries:

```
python src/single_seq_context.py  -i <INPUT FILE>\
 --dict_path <FAMILY DICT FOLDER> --uniref50_path <UNIREF50 DICT FILE>
```

An example input file with 188 sequences selected from Deep Mutational Scanning (DMS) dataset is included in `dataset/DMS_proteingym.fasta`. The Pfam family dictionaries and UniRef50 general dictionaries can be accessed per request, and a download site for open access is currently under construction.

## Constructing Your Own Dictionary

To construct a dictionary based on your own homologous sequence database, first run the following command to generate the raw word by:

```
python src/main.py -i <INPUT FILE> --mode batch -o <RAW WORD OUTPUT PATH>
```

In this step, a `pandas` DataFrame including the metadata of all raw words extracted from the sequences is generated.
Next, run the following command to process it, extract high-frequency protein words and build a word list from the raw words:

```
python src/table_merger.py  --prefix <RAW WORD FILE> --out_dir <OUTPUT PATH>
```

This step will finally generate a dictionary file in `numpy` .npy format. It will also produce a filtered word list of your input sequence containing high-frequency protein words only. 


# Paper

[Automatically Defining Protein Words for Diverse Functional Predictions Based on Attention Analysis of a Protein Language Model](https://www.biorxiv.org/content/10.1101/2025.01.20.633699v1)

# License

This project is covered under the **Apache 2.0 License**.
