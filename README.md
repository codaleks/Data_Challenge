# 🧠 Fair Text Classification

This project addresses a multi-class text classification task with a focus on fairness. The objective is to accurately assign one of 28 possible categories to each text, while ensuring that the model remains fair across sensitive demographic groups, such as gender.

---

## 📘 Task Overview

The task is straightforward:

> Given a text document (represented by its embedding), predict its category among **28 possible classes**.

This is a **multi-class classification problem**, where the input is a feature vector (embedding), and the output is one of the 28 class labels.

We follow a common paradigm in NLP:
1. **Document representation**: Each text is represented as a fixed-length **feature vector** (embedding), directly provided in the dataset (pre-computed using BERT).
2. **Classification**: A model is trained to predict the document's label from its feature vector.

---

## 🎯 Goal of the Challenge

The primary objective is two-fold:

1. **Accuracy** – Build a model that predicts the correct category with high precision.
2. **Fairness** – Ensure that the model does not exhibit **bias** toward or against particular demographic groups (e.g., gender).

The evaluation procedure (explained in the notebook) includes both traditional metrics (e.g., accuracy) and fairness-oriented metrics.


