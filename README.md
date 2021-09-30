# Structural Text Segmentation of Legal Documents
#### by Dennis Aumiller*, Satya Almasian*, Sebastian Lackner and Michael Gertz
*Equal Contribution.

---
## Online Models

You can now directly load the PyTorch weights in Huggingface transformers for the Model Roberta CLS consec (run 5): https://huggingface.co/dennlinger/roberta-cls-consec. This was the best-performing run on base transformers (not Sentence-transformers).  
**Update 2021-09-30:** We have now a model trained on Wikipedia paragraphs instead, thanks in large parts to Lucienne-Sophie Marmé. The model can be found online on the Huggingface model hub: https://huggingface.co/dennlinger/bert-wiki-paragraphs

---

This repository contains scripts to reproduce the results of the paper https://arxiv.org/abs/2012.03619, for transformer based models for text segmentation. 

This repository contains mostly data generation and evaluation scripts. For pre-trained models used in the paper, please reach out to the authors via `<lastname>@informatik.uni-heidelberg.de`.

## Dataset
Our Terms-of-Service dataset is publicly available at https://heibox.uni-heidelberg.de/f/749127d934cb4a64929c/?dl=1. 
For more information on the data please refer to the paper and appendix. 
The dataset consist of `json` files for Terms of Services crawled from the web. 
An example file structure can be seen here: 
```
{
    "text": "a paragraph of the terms of services",
    "section": [list of all the headings and subheading that the paragraph belongs to]
  },
  {....
  
  }
```

## Training from Scratch
This repository does not include training scripts to train the models on your own data: 

* __sentence-transformer__: please refer to our forked project [sentence-transformer](https://github.com/dennlinger/sentence-transformers). 
All the training scripts for baselines, such as Bag of Words, average of GloVe and tf-idf can be found in the `segmenation_baseline` folder. 
The Sentence-Transformer-based RoBERTa and BERT can be trained using the scripts in `segmenation_models`.
* __transformer-language models__: please refer to our forked project [transformers](https://github.com/dennlinger/transformers).
* __GraphSeg__: The GraphSeg model is available in Java and can be accessed [here](https://bitbucket.org/gg42554/graphseg/src/master/).
For our evaluation we ran the standalone binary provided in the repository, with a minimum section length of two sentences,
and a segmentation threshold of `0.1`-`0.5` in increments of `0.1`.
* __WikiSeg__: Implementation of the paper `Text Segmentation as a Supervised Learning Task` 
can be accessed [here](https://github.com/koomri/text-segmentation). Note that this project has different requirements.


## Requirements
Our code has been run on both `python 3.7` and `python 3.8`,
but in order to run the evaluation script for the baseline model of textseg you need to switch to `python 2`, 
since their model is trained and loaded with `python 2` packages. 
Basically, for the script `03_test_textseg` to work, you need to switch to the environment 
from the [WikiSeg repository](https://github.com/koomri/text-segmentation).
For the remaining code, please refer to our requirements: 
```
pip install -r requirements.txt
```

## Usage
### Step one (data cleaning): 

After downloading the dataset, you should start by cleaning the files and filtering them to the relative topics, 
using `our_model/01_clean_sections.py`. This script takes the raw data and filters out the irrelevant sections and paragraphs.
The topics and the heading associations are taken from a dictionary in `our_model/utils.py`. Example:
```bash
python3 our_model/01_clean_sections.py --paragraph  True --input_folder INPUT_FOLDER --output_folder OUTPUT_FOLDER                   
```
if `paragraph` is set to `True`, then the paragraph structure is kept. Otherwise they are merged into a single section. 
`input_folder` is the location of the terms of services dataset.

### Step two (generating training data): 
To generate training pair with different training strategies mentioned in the paper, 
use `our_model/02_generate_training_pairs.py`. For the baselines there are also respective scripts in:
`baseline/02_generate_training_data_graphseg.py` and  `baseline/02_generate_training_data_textseg.py`. 
Example:
```bash
python our_model/02_generate_training_pairs.py --input_folder INPUT_FOLDER --output_folder OUTPUT_FOLDER \
  --sentence_tokenizer_path SENTENCE_TOKENIZER_PATH \
  --sample_number 3 \
  --heading_level 1 \
  --storage_method raw \
  --random_split False \
  --consecutive True 
```
`INPUT_FOLDER` is the cleaned dataset from the previous step. 
In case you want to pre-process the data with sentencepiece tokenizer you need to set the path in `sentence_tokenizer_path`. 
'sample_number' is the number of positive and negative samples, our default is 3. 
`heading_level` allows to choose between the first and second level headings in the data, 
all the models from the paper were trained using the first level heading. 
`storage_method` allows for further pre-processing by tokenizing based on bert, roberta or sentence-piece tokenizer. 
Most models in the paper were using the `raw` format. 
If `random_split` is applied then all the paragraph and sections are mixed and then randomly shuffled for the test and train set. 
If `consecutive` is set then the positive examples come from the paragraphs belonging to the same section.
In the paper, we discuss three different training strategies. 
For Section Prediction, in the previous step the `paragraph` flag has to be set to `False` and the `random_split` set to `True`,
 also `consecutive` set to `False`. 
 For Random Paragraph, in the previous step the `paragraph` flag has to be set to `True` and the `random_split` set to `True`, 
 also `consecutive` set to `False`. 
 For Consecutive Paragraph, in the previous step the `paragraph` flag has to be set to `True` and the `random_split` set to `False`, 
 also `consecutive` set to `True`.

As for the baselines, we have two separate scripts to transform our data to the valid format for the models, example: 
```bash
python baselines/02_generate_training_data_graphseg.py --input_folder INPUT_FOLDER --output_folder OUTPUT_FOLDER
``` 

```bash
python baselines/02_generate_training_data_textseg.py --input_folder INPUT_FOLDER --output_folder OUTPUT_FOLDER --heading_level 1
```

where the `INPUT_FOLDER` contains the cleaned data from step one.

### Step three (training): 
For training we refer to the Training from Scratch section, where the code for each model can be found. 
We also have some helper script for the GraphSeg algorithm, as it usually crashes when the paragraphs are too small. 
To resume training you can use: 
```bash
python baselines/03_continue_graphseg_training.py --existing_dir EXISTING_DIR \
--new_dir NEW_DIR \
--so_far_completed_dir SO_FAR_COMPLETED_DIR
```
where the `EXISTING_DIR` contains all the data and `NEW_DIR` is the path to save the resumed data files, 
`SO_FAR_COMPLETED_DIR` contains the files that have already been processed. 

### Step four and five (evaluation): 
The prediction accuracy for Same Topic Prediction task is reported during training and final evaluation according to 
the scripts available in the forked repositories. For the text segmentation task, the code to generate the plots and 
results reported in the paper is in `our_model/05_eval_ensemble.py`. 
To run the code successfully you need the trained models from the transformed based, sentence-transformer and all the baselines.
For GraphSeg and WikiSeg, we had to modify their output to be comparable to our results. 
GraphSeg's default output format already indicates the position of separations, 
we use `baselines/04_backtranslate_seg_results.py` to convert this result to a comparable paragraph-based prediction. 
The same applies to textseg, `baselines/04_test_textseg.py` provides code for loading a trained model and making segment prediction based on it.
This script operates in the textseg environment which is different form the rest of the repository. 
The output of this script has to be further processed by `baselines/04_backtranslate_seg_results.py` to achieve comparable results.

Example of the back translation: 
```bash
python baselines/04_backtranslate_seg_results.py --input_folder INPUT_FOLDER \
--seg_folder SEG_FOLDER \
--output_folder OUTPUT_FOLDER \
--method {textseg,graphseg}
```
where the `method` parameter defines whether to use WikiSeg (sometimes referred to as `textseg`) or GraphSeg, 
`SEG_FOLDER` must contain the segmented text from the models and the `INPUT_FOLDER` the original test data.  

## Citation
If you use our code, dataset, or model weights in your research, please cite.


## License
[MIT](https://choosealicense.com/licenses/mit/)
