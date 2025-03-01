from datasets import load_dataset

ds = load_dataset("gsm8k", "main")


def getTestDataPrefix(howmany=10):
    questions = []
    answers = []

    for i in range(howmany):
        questions.append(ds["test"][i]["question"])
        answers.append(ds["test"][i]["answer"])

    return questions, answers
