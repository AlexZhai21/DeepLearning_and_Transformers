from datasets import load_dataset
import pandas as pd


def load_oasst1():
    return load_dataset("OpenAssistant/oasst1")


def load_oasst1_examples(split="train"):
    ds = load_oasst1()
    df = pd.DataFrame(ds[split])

    df = df[
        (df["deleted"] == False)
        & (df["lang"] == "en")
        & (df["text"].notna())
    ].copy()

    by_id = df.set_index("message_id")
    examples = []

    for _, row in df.iterrows():
        if row["role"] != "assistant":
            continue

        parent_id = row["parent_id"]
        if parent_id not in by_id.index:
            continue

        parent = by_id.loc[parent_id]
        if parent["role"] != "prompter":
            continue

        user_text = parent["text"].strip()
        assistant_text = row["text"].strip()

        if not user_text or not assistant_text:
            continue

        examples.append({
            "user": user_text,
            "assistant": assistant_text,
        })

    return examples
