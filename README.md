# The QA-SRL Gold Standard
A repository for high-quality QASRL data collected from crowd-workers.
This repository is the reference point for the dataset and evaluation protocols described in the paper _Controlled Crowdsourcing for High-Quality QA-SRL Annotation_.

* The paper can be found [here](https://www.aclweb.org/anthology/2020.acl-main.626/)
* The data files are located [here](data/) 
* The evaluation code can be found [here](scripts/)

## Paper Abstract
Question-answer driven Semantic Role Labeling (QA-SRL) was proposed as an attractive
open and natural flavour of SRL, potentially attainable from laymen. Recently, a large-scale
crowdsourced QA-SRL corpus and a trained parser were released. Trying to replicate the
QA-SRL annotation for new texts, we found that the resulting annotations were lacking in
quality, particularly in coverage, making them insufficient for further research and evaluation.
In this paper, we present an improved crowdsourcing protocol for complex semantic annotation, involving worker selection and training, and a data consolidation phase. Applying this protocol to QA-SRL yielded highquality annotation with drastically higher coverage, producing a new gold evaluation dataset.
We believe that our annotation protocol and gold standard will facilitate future replicable
research of natural semantic annotations. 

## Data Files
The data files are organized based on the source corpora: (1) Wikinews and (2) Wikipedia and the development and test partitions. The sentences were sampled from the large-scale dataset created by [Fitzgerald, 2018](https://www.aclweb.org/anthology/P18-1191.pdf), with 1000 sentences from each source split equally between development and test. 

* The sentences can be found under [data/sentences](data/sentences)
* The expert set described in the paper can be found under [data/expert](data/expert)
* The QA-SRL gold annotation (by a pipeline of 3 trained workers) can be found under [data/gold](data/gold)
* Evaluation scripts can be found under [qasrl/](qasrl/) folder. See the next sections on how to apply the evaluation procedure.

## Data Format
The data is presented in tabular, comma separated format, conversion to the data format used in [Large-Scale QASRL](www.qasrl.org) is underway.
The CSV format includes the following headers:
1. qasrl_id - Sentence identifier. Same id is used in the sentence files.
2. verb_idx - Zero-based index of the predicate token
3. verb - The verb token as appearns in the sentence in token *verb_idx*
4. question - The question representing the role.
5. answer - Multiple answer spans, separated by: ~!~. Each answer is a contigious span of tokens that depicts an argument for the role.
6. answer_range - Multiple token ranges, separated by: ~!~. Each range corresponds to the answer span in the same position in the answer column, and is formatted with INCLUSIVE_START:EXCLUSIVE_END token indices from the sentence.

Fields 7 through 14 are taken from the QA-SRL question template, as parsed by the QASRL state-machine. Given a _valid_ QASRL question you can re-run the state machine using [a sample script from the annotation repository](https://github.com/plroit/qasrl-crowdsourcing/blob/ecbplus/qasrl-crowd-example/jvm/src/main/scala/example/RunQuestionParser.scala) and re-create these fields together with some more data. 

7. wh - The WH question word
8. aux - The Auxilliary slot
9. subj - The subject placeholder (someone or something)
10. obj - The direct object placeholder (someone or something)
11. prep - The preposition used for the indirect object 
12. obj2 - The indirect object placeholder (someone or something)
13. is_negated - Boolean, detects if there is a negation in the question
14. is_passive - Boolean, detects if passive voice is used in the question

## Evaluating QA-SRL system output.
To evaluate a QA-SRL system output against a reference QASRL data you will have to follow these instructions.
1. Compile both datasets into the described CSV format.
2. Use the script [evaluate_dataset.py](scripts/) with the following command line arguments:
  1. Path to the system output CSV file
  2. Path to the reference (ground truth) CSV file
  3. Path to the sentences file (optional) to create a complete matched/unmatched table
