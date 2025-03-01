from datasets import load_dataset
import pandas as pd


def getTestDataPrefix(howmany=10):
    ds = load_dataset("gsm8k", "main")
    questions = []
    answers = []
    num_data = len(ds["test"])

    for i in range(num_data):
        questions.append(ds["test"][i]["question"])
        answers.append(ds["test"][i]["answer"])
    # Save questions and answers to CSV
    df = pd.DataFrame({"question": questions, "answer": answers})

    # Save to CSV file
    df.to_csv("gsm8k_test_data.csv", index=False)

    return questions, answers


getTestDataPrefix()
