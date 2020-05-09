# retroqa
CIS 700 (Prof. Dan Roth) Final Project - Reasoning for Natural Language Understanding

This repo contains our code for the "Retrospective Reading and Dynamic Memory Networks for Question Answering" project.

Abstract: In recent years, memory networks have seen great success in the task of question answering (QA) over unstructured text. Our work builds on previous applications of memory networks in two major ways. First, we augment the state-of-the-art model on the SQuAD dataset with a Dynamic Memory Network (DMN+) module, which enhances the model with explicit reasoning capabilities. We also add a verification module in order to ensure the quality of the selected answer and to account for unanswerable questions. Evaluating on the [Microsoft NewsQA](https://www.microsoft.com/en-us/research/project/newsqa-dataset/) dataset, we observe that our changes facilitate the reasoning process and improve the performance of the model in several key aspects.

Model architecture:

![Model architecture](model.png)

To run our code, please run the run.py script. This will load the spaCy tokenized versions of the [training](https://drive.google.com/drive/folders/1lupUpHV4hjzV3bzK3CaKYzl1kUMCt46f) and [validation](https://drive.google.com/drive/folders/1lupUpHV4hjzV3bzK3CaKYzl1kUMCt46f) sets and perform the experiments.
