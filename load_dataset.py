from datasets import load_dataset


def load_math_dataset():
    questions = []
    answers = []
    # Load the mathematics dataset from HuggingFace
    ds = load_dataset("HuggingFaceH4/MATH-500")
    for i in range(len(ds["test"][0])):
        questions.append(ds["test"][i]["problem"])
        answers.append(ds["test"][i]["answer"])
    return questions, answers


def save_to_csv():
    questions, answers = load_math_dataset()
    import pandas as pd

    df = pd.DataFrame({"question": questions, "answer": answers})
    df.to_csv("math_test_data.csv", index=False)


save_to_csv()
