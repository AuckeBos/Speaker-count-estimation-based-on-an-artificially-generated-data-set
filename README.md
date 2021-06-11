# Speaker count estimation based on an artificially generated data set

This repo is created for the exam project of the
course [Automatic Speach Recognition](https://www.ru.nl/courseguides/socsci/courses-osiris/ai/let-rema-lcex10-automatic-speech-recognition/) (
LET-REMA-LCEX10). The goal, its structure, and its results are elaborated upon in the corresponding [paper](Paper.pdf).

In short: the goal of the project is to try to train a Recurrent Neural Network on an artifically created dataset of
concurrent speakers. We tested performance on different levels.

# Installation

* Install [Poetry](https://python-poetry.org/): `pip install poetry`
* Run `poetry install` to install the virtual environment
* Run `poetry python main.py` to run the project. Choose which tasks to run by changing the variable at the top of the
  file.
* \[Optional\] Run `poetry shell` to enter the virtual environment