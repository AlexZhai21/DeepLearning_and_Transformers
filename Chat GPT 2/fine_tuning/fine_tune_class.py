import torch
from torch.utils.data import Dataset
import tiktoken

class ChatSFTDataset(Dataset):
    def __init__(self, examples, block_size=1024):
        """
        examples: list of dicts like:
        {
            "user": "...",
            "assistant": "..."
        }
        """
        self.examples = examples
        self.block_size = block_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc.eot_token

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        user_text = ex["user"].strip()
        assistant_text = ex["assistant"].strip()

        # Prefix is context. We do NOT train loss on this.
        prefix = f"User: {user_text}\nAssistant:"

        # Answer is what we DO train loss on.
        answer = f" {assistant_text}<|endoftext|>"

        prefix_ids = self.enc.encode(prefix)
        answer_ids = self.enc.encode(
            answer,
            allowed_special={"<|endoftext|>"}
        )

        full_ids = prefix_ids + answer_ids

        # truncate
        full_ids = full_ids[: self.block_size + 1]

        # x predicts y shifted by one
        x = torch.tensor(full_ids[:-1], dtype=torch.long)
        y = torch.tensor(full_ids[1:], dtype=torch.long)

        # Mask all labels whose target token is still inside the prefix.
        # y[i] corresponds to full_ids[i+1].
        prefix_len = len(prefix_ids)

        for i in range(len(y)):
            target_position = i + 1
            if target_position < prefix_len:
                y[i] = -100

        # pad to block_size
        pad_len = self.block_size - len(x)
        if pad_len > 0:
            x = torch.cat([
                x,
                torch.full((pad_len,), self.enc.eot_token, dtype=torch.long)
            ])
            y = torch.cat([
                y,
                torch.full((pad_len,), -100, dtype=torch.long)
            ])

        return x, y