import pandas as pd
import re
import torch
import torch.nn as nn
# FIX 1a: Use AutoTokenizer/AutoModel so we can load BERTweet (vinai/bertweet-base)
# BERTweet is pretrained on 850M tweets — much better for noisy social media text
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# STEP 1: Load TRAIN dataset
df = pd.read_csv("train_data_cleaning2.csv")
df = df[['text', 'target']].dropna()

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# STEP 2: Clean text
def clean(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    return text.lower()

train_df['text'] = train_df['text'].apply(clean)
test_df['text'] = test_df['text'].apply(clean)

# STEP 3: Temporal Windowing
# FIX 2: W=10, S=5 with 50% overlap -- matches Section 3.3.1 (Equation 9)
#
# HOW IT WORKS:
#   window_size (W) = 10  -> each window contains 10 posts
#   stride      (S) = 5   -> we move forward 5 posts each time (50% overlap)
#
# Example with 20 posts:
#   Window 0 -> posts [0..9]
#   Window 1 -> posts [5..14]   <- posts 5-9 appear AGAIN (overlap)
#   Window 2 -> posts [10..19]
#
# Why this matters: a crisis signal at post 7 now appears in BOTH window 0
# and window 1, giving the model two chances to catch it early (EDR@1).
#
# Label: majority voting -- if >50% of posts in a window are crisis,
# the whole window is labelled crisis. This matches Equation 2.

def create_windows(texts, labels, window_size=10, stride=5):
    windowed_texts  = []
    windowed_labels = []
    windowed_deltas = []   # FIX 3: one delta sequence per window

    # stop early enough so we never slice out of bounds
    for i in range(0, len(texts) - window_size + 1, stride):
        window_texts  = texts[i : i + window_size]
        window_labels = labels[i : i + window_size]

        # join all posts in the window into one string for tokenisation
        text_window  = " ".join(window_texts)

        # majority voting: round(mean) -> 1 if >50% are crisis, else 0
        label_window = round(sum(window_labels) / len(window_labels))

        # FIX 3: Time-delta encoding -- Equation 3 / Section 3.3.2
        # We use post index as a proxy for time (no real timestamps in this dataset).
        # delta_t = (index - first_index) / (last_index - first_index + eps)
        # This gives each post a normalised position 0.0 -> 1.0 within the window.
        eps          = 1e-8
        first_idx    = float(i)
        last_idx     = float(i + window_size - 1)
        deltas       = [(float(i + t) - first_idx) / (last_idx - first_idx + eps)
                        for t in range(window_size)]   # list of W floats in [0, 1]

        windowed_texts.append(text_window)
        windowed_labels.append(label_window)
        windowed_deltas.append(deltas)

    return windowed_texts, windowed_labels, windowed_deltas

# APPLY WINDOWING -- window_size=10, stride=5 match Table 3 selected values
train_texts, train_labels, train_deltas = create_windows(
    train_df['text'].tolist(),
    train_df['target'].tolist()
)

test_texts, test_labels, test_deltas = create_windows(
    test_df['text'].tolist(),
    test_df['target'].tolist()
)

# Sanity check -- verify windowing worked correctly
print(f"Training windows : {len(train_texts)}")
print(f"Test windows     : {len(test_texts)}")
print(f"Window label split -- crisis: {sum(train_labels)}, "
      f"non-crisis: {len(train_labels) - sum(train_labels)}")
print(f"Sample delta sequence (window 0): {[round(d,2) for d in train_deltas[0]]}")
# STEP 4: Tokenization
# FIX 1a: BERTweet tokenizer — handles tweet-specific tokens like URLs, mentions, emojis
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', use_fast=False)

train_tokens = tokenizer(
    train_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

test_tokens = tokenizer(
    test_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# STEP 5: Create datasets
# FIX 3: Add delta tensors -- shape (num_windows, window_size)
# Each row is the list of 10 time-delta values for one window
train_delta_tensor = torch.tensor(train_deltas, dtype=torch.float32)
test_delta_tensor  = torch.tensor(test_deltas,  dtype=torch.float32)

train_dataset = TensorDataset(
    train_tokens['input_ids'],
    train_tokens['attention_mask'],
    torch.tensor(train_labels),
    train_delta_tensor             # FIX 3: deltas travel with each sample
)

test_dataset = TensorDataset(
    test_tokens['input_ids'],
    test_tokens['attention_mask'],
    torch.tensor(test_labels),
    test_delta_tensor              # FIX 3
)

# Reduced batch size to 8 — gives more gradient updates per epoch on small dataset
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# STEP 6: Model
class CrisisModel(nn.Module):
    def __init__(self):
        super().__init__()
        # FIX 1a: BERTweet encoder
        self.bert = AutoModel.from_pretrained('vinai/bertweet-base')

        # FIX 3: LSTM input is now 769 (768 BERT dims + 1 time-delta dim)
        # This matches Equation 3: e~t = [et || delta_t] in R^(d+1)
        self.lstm      = nn.LSTM(769, 256, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(512, 1)
        # FIX 5: dropout p=0.3 before classification head (Section 3.4.3)
        self.dropout   = nn.Dropout(p=0.3)
        # TWO-LAYER FFN HEAD matching Equation 7: y^ = sigmoid(W2 * ReLU(W1*c + b1) + b2)
        # 512 -> 256 -> 1
        self.fc1       = nn.Linear(512, 256)
        self.fc2       = nn.Linear(256, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, deltas):
        # deltas shape: (batch, window_size) -- one scalar per post in the window
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state                        # (batch, seq_len, 768)

        # FIX 3: Concatenate delta onto every token position in the sequence
        # deltas is (batch, W=10); we need (batch, seq_len, 1) to concat with BERT output.
        # We repeat the W delta values across all seq_len token positions.
        # Specifically, we tile the window delta vector to match seq_len by interpolation:
        # simplest approach -- repeat each delta value across (seq_len // W) tokens,
        # then trim/pad to exactly seq_len.
        batch_size, seq_len, _ = bert_out.shape
        W = deltas.shape[1]                        # window_size = 10

        # Expand deltas: (batch, W) -> (batch, seq_len)
        # Each of the W posts contributes (seq_len/W) tokens on average
        delta_expanded = deltas.unsqueeze(2)       # (batch, W, 1)
        # Interpolate W steps -> seq_len steps using linear interpolation
        delta_seq = torch.nn.functional.interpolate(
            delta_expanded.permute(0, 2, 1),       # (batch, 1, W)
            size=seq_len,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1)                         # (batch, seq_len, 1)

        # Concatenate: (batch, seq_len, 768) || (batch, seq_len, 1) = (batch, seq_len, 769)
        x = torch.cat([bert_out, delta_seq], dim=2)

        lstm_out, _  = self.lstm(x)                # (batch, seq_len, 512)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context      = torch.sum(attn_weights * lstm_out, dim=1)

        # FIX 5: dropout then two-layer FFN head (Equation 7)
        context = self.dropout(context)
        out     = self.fc2(torch.relu(self.fc1(context)))
        return self.sigmoid(out)

# STEP 7: Train
model = CrisisModel()

# FIX 1b: Weighted BCE Loss
# The paper uses class weights inversely proportional to frequency (Section 3.2.3)
# This stops the model from lazily predicting "not crisis" for everything
num_crisis     = sum(train_labels)          # count of positive (crisis) samples
num_non_crisis = len(train_labels) - num_crisis
total          = len(train_labels)
w_pos = total / (2 * num_crisis)            # w+ = total / (2 * n_positive)
w_neg = total / (2 * num_non_crisis)        # w- = total / (2 * n_negative)
print(f"Class weights → w+={w_pos:.3f}  w-={w_neg:.3f}")

def weighted_bce(outputs, targets):
    """Weighted binary cross-entropy matching Equation 8 in the paper."""
    # FIX: ensure both are always 1D — handles last batch with 1 sample
    # where squeeze() turns shape [1] into scalar []
    outputs = outputs.view(-1)
    targets = targets.view(-1)
    weights = torch.where(targets == 1,
                          torch.tensor(w_pos),
                          torch.tensor(w_neg))
    bce = nn.BCELoss(weight=weights)
    return bce(outputs, targets)

# FIX 1c: AdamW optimizer (Adam + proper weight decay — prevents BERT overfitting)
# lr=5e-4 matches Table 3 selected value; weight_decay=1e-2 matches Section 3.5.2
# Lowered lr from 5e-4 to 2e-5 — standard for BERT fine-tuning, prevents overshooting
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

# FIX 1c: Cosine annealing scheduler with linear warmup (Section 3.5.2)
# Warmup over first 10% of steps, then cosine decay — stabilises BERT fine-tuning
from transformers import get_cosine_schedule_with_warmup
num_epochs   = 5  # increased from 2 — model needs more passes to learn
total_steps  = num_epochs * len(train_loader)
warmup_steps = int(0.1 * total_steps)       # 10% warmup as stated in paper
scheduler    = get_cosine_schedule_with_warmup(optimizer,
                                               num_warmup_steps=warmup_steps,
                                               num_training_steps=total_steps)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    model.train()
    for input_ids, mask, labels, deltas in train_loader:   # FIX 3: unpack deltas
        outputs = model(input_ids, mask, deltas).squeeze() # FIX 3: pass deltas
        loss = weighted_bce(outputs, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print("Loss:", loss.item())

# STEP 8: Evaluate
preds = []
true = []

model.eval()
with torch.no_grad():
    for input_ids, mask, labels, deltas in test_loader:    # FIX 3: unpack deltas
        outputs = model(input_ids, mask, deltas).squeeze() # FIX 3: pass deltas
        predictions = (outputs > 0.5).int().view(-1)
        preds.extend(predictions.tolist())
        true.extend(labels.tolist())

print("Accuracy:", accuracy_score(true, preds))
print("F1 Score:", f1_score(true, preds))

# Confusion Matrix
cm = confusion_matrix(true, preds)
print("Confusion Matrix:\n", cm)

# ── STEP 8b: Early Detection Metrics ─────────────────────────────────────────
# FIX 4: EDR@k and Mean Time-to-Detection (TTD)
# These are the metrics that distinguish this paper from static classifiers.
# They measure HOW EARLY the model catches a crisis, not just whether it does.
#
# Since our dataset has no real event boundaries, we simulate events by
# grouping consecutive windows into blocks of EVENT_SIZE windows each.
# Within each block, we check at which window the model first fires a crisis alert.
#
# EDR@k (Equation 13): fraction of crisis events detected within first k windows
# TTD   (Equation 14): average number of windows until first correct crisis alert

EVENT_SIZE = 5   # treat every 5 consecutive test windows as one "event"

# Split predictions and ground truth into event-sized groups
events_pred = [preds[i : i + EVENT_SIZE] for i in range(0, len(preds), EVENT_SIZE)]
events_true = [true[i  : i + EVENT_SIZE] for i in range(0, len(true),  EVENT_SIZE)]

# Only evaluate events that actually contain at least one crisis window
crisis_events_pred = []
crisis_events_true = []
for ep, et in zip(events_pred, events_true):
    if 1 in et:                          # this event has at least one crisis window
        crisis_events_pred.append(ep)
        crisis_events_true.append(et)

total_crisis_events = len(crisis_events_pred)
print(f"\nTotal crisis events for EDR evaluation: {total_crisis_events}")

# For each crisis event, find the index of the first window where model fires alert
# -1 means the model never detected this event at all
def first_detection(pred_windows):
    for i, p in enumerate(pred_windows):
        if p == 1:
            return i      # 0-indexed: 0 = detected on first window (best)
    return -1             # never detected

detection_steps = [first_detection(ep) for ep in crisis_events_pred]

# EDR@k: fraction of events detected at step <= k-1 (k is 1-indexed)
def edr_at_k(detection_steps, k):
    detected = sum(1 for s in detection_steps if 0 <= s <= k - 1)
    return detected / len(detection_steps) if detection_steps else 0.0

edr1 = edr_at_k(detection_steps, 1)
edr3 = edr_at_k(detection_steps, 3)
edr5 = edr_at_k(detection_steps, 5)

print(f"EDR@1 : {edr1:.3f}  (detected on very first window)")
print(f"EDR@3 : {edr3:.3f}  (detected within first 3 windows)")
print(f"EDR@5 : {edr5:.3f}  (detected within first 5 windows)")

# TTD: mean number of windows to first correct detection (ignoring missed events)
detected_steps = [s for s in detection_steps if s >= 0]
ttd = sum(detected_steps) / len(detected_steps) if detected_steps else float('inf')
missed = detection_steps.count(-1)

print(f"Mean TTD      : {ttd:.2f} windows  (lower = faster detection)")
print(f"Missed events : {missed} / {total_crisis_events}")
print(f"\nPaper targets: EDR@1=0.68, EDR@3=0.84, EDR@5=0.92, TTD~6.5 windows")

# STEP 9: Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved!")

# STEP 10: External dataset (REAL-WORLD TEST)
external_test_df = pd.read_csv("test_data_cleaning2.csv")
external_test_df = external_test_df[['text']].dropna()
external_test_df['text'] = external_test_df['text'].apply(clean)

sample_texts = external_test_df['text'].tolist()[:5]

# FIX: repeat each text 10 times to simulate a window of 10 posts
# This matches the training format (windows of 10 joined tweets)
W = 10
windowed_sample_texts = [" ".join([t] * W) for t in sample_texts]

inputs = tokenizer(windowed_sample_texts, return_tensors='pt', padding=True,
                   truncation=True, max_length=128)

n = len(windowed_sample_texts)
ext_deltas = torch.tensor(
    [[t / (W - 1) for t in range(W)] for _ in range(n)],
    dtype=torch.float32
)

outputs = model(inputs['input_ids'], inputs['attention_mask'], ext_deltas)

print("\n--- External Dataset Predictions ---")
for text, pred in zip(sample_texts, outputs):
    label = "Crisis" if pred.item() > 0.5 else "Not Crisis"
    print(f"\nText: {text}\nPrediction: {label} (confidence: {pred.item():.3f})")