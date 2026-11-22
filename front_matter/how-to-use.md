# How to Use This Book

## Anatomy of a chapter

Each technical chapter follows the same structure, so that you always know where to look:

1. **Learning objectives and intuition.** What you will be able to do after the chapter, and the main idea in plain words.
2. **Mathematical formulation and derivations.** The definitions and equations, with each step explained.
3. **Figures.** Illustrations produced by Python code, so that you can reproduce and modify them.
4. **Hands-on implementation.** The method implemented and applied to a real dataset.
5. **Worked examples.** Problems solved completely, first by hand and then verified with Python.
6. **Exercises.** Problems for you to solve, split into manual exercises and Python exercises.

Short boxes appear throughout the text:

- **Key idea:** the central point of a section, stated briefly.
- **Pitfall:** a common mistake, and how to avoid it.
- **Try it:** a small experiment to run in the notebook before moving on.

## Reading and running the notebooks

Each technical chapter is a Jupyter notebook that combines text, equations, code, outputs, and figures. There are two ways to use it:

- **Read it online.** The published website contains the chapters with all outputs and figures already computed, so you can read the book without installing anything.
- **Run it yourself.** Clone the repository, create the environment, and execute the notebooks on your own computer. Chapter 1 explains the setup step by step.

A few conventions make the results reproducible:

- Random operations use a fixed seed, so that your results should match those printed in the book when you use the same versions of the libraries.
- Datasets are loaded with library functions rather than from files you have to find. Some of them, for example the image datasets of torchvision, are downloaded automatically the first time you run a notebook, so you need an internet connection once.
- The notebooks are designed to run on an ordinary computer. A GPU makes the deep learning chapters faster but is not required.

## Exercises

Exercises come in two kinds:

- **Manual (analytical) exercises** are solved with pen and paper. They train you to derive, compute, and reason.
- **Python (computational) exercises** are solved on the computer, usually with a real dataset. They train you to implement, experiment, and interpret.

Each exercise is marked with a difficulty level:

- **★ Warm-up:** checks that you understood a definition or a formula.
- **★★ Core:** applies the chapter's ideas to a new but similar situation.
- **★★★ Challenge:** combines several ideas or asks you to explore. It is meant to be demanding, not impossible.

Exercises are different from the worked examples, which are solved completely in the text. For each exercise you will find hints after the statement, and complete solutions or detailed solution outlines are collected in Appendix C. Try the exercise before reading the hints, and read the hints before looking at the solution.

## Suggestions for studying

- **Derive once by hand.** Follow each derivation on paper at least once. Writing the steps yourself is the best way to understand them.
- **Predict, then run.** Before executing a cell, guess what it will output. When the result surprises you, you have found something worth understanding.
- **Change things.** Modify a hyperparameter, a dataset, or a line of code, and observe the effect. The notebooks are meant to be altered.
- **Move between math and code.** When you read a formula, find the line of code that implements it, and the other way around.
