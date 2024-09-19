# Tools 

This directory contains the programs to support ALPR system development. This includes the following functionalities: 
- `tools/data`: Programs for data preprocessing and generation. 
- `tools/sdk`: Development toolkits. 
- Add more here...


## Data 
The following programs are included in the `tools/data` directory. 
### 1. Generate Synthetic Data 
Please refer to `tools/data/gensynth.py` for code definition. Example usage:
```bash 
python gensynth.py \
    --spec ../../configs/ALPR-data.yaml \
    --max-objs 4 \
    --max-lines 3 \
    --num-train 2000 \
    --num-val 500 \
    -result-dir results/synth-data-output
```