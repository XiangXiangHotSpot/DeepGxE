# DeepGxE
This repository provides the code and dataset processing pipeline for the paper “Large-scale crop dataset and deep learning-based multi-modal fusion framework for G×E genomic prediction”

## Step 1: Download the Raw Data

1. Phenotype Data: All the phenotypic records of the 1st to 6th WYCYT cycles, 11th to 27th HRWYT cycles, 24th to 39th ESWYT cycles and 36th to 52nd IBWSN cycles can be searched at [https://data.cimmyt.org/dataverse/cimmytdatadvn](https://data.cimmyt.org/dataverse/cimmytdatadvn).

2. Genotype Data: The genotypic data used in this study can be obtained from the CIMMTY dataverse: [https://hdl.handle.net/11529/10695](https://hdl.handle.net/11529/10695).  
   *(F_MAF0.01_Miss50_Het10-Merged.all.discover.lines.and.selection.candidates.vcf.imputed.CIMMYT.2022.hmp.txt.gz)*

3. The environmental data from the AgERA5 dataset are publicly available and can be directly downloaded at [https://hdl.handle.net/11529/10548548](https://hdl.handle.net/11529/10548548).

## Step 2: Dataset Orgnization

Follow the guidelines provided in PlantHeightProcessing and WheatYieldProcessing to prepare a comprehensive “Genotype-Environment-Phenotype” dataset for wheat yield and plant height traits.

## Step 3: Model Training and Testing
To train and test the model, simply run the following commands:

**Using Python Scripts**

```bash
cd Bi-LSTM/
python run.py
```

```bash
cd Transformer/
python run.py
```

**Using Shell Scripts**
```bash
cd Bi-LSTM/
sh run.sh
```

```bash
cd Transformer/
sh run.sh
```
