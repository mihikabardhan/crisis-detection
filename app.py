import streamlit as st
import torch
import torch.nn as nn
import re
# FIX 1a: Use AutoTokenizer/AutoModel to load BERTweet
from transformers import AutoTokenizer, AutoModel

# Page config
st.set_page_config(page_title="Crisis Prediction System", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stTextArea textarea {
        background-color: #262730;
        color: white;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🚨 Crisis Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect crisis-related tweets using AI</p>", unsafe_allow_html=True)

# ✅ CLEAN FUNCTION (IMPORTANT)
def clean(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    return text.lower()

# Load tokenizer
# FIX 1a: BERTweet tokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', use_fast=False)

# Model
class CrisisModel(nn.Module):
    def __init__(self):
        super().__init__()
        # FIX 1a: BERTweet encoder
        self.bert    = AutoModel.from_pretrained('vinai/bertweet-base')
        # FIX 3: LSTM input is 769 (768 BERT dims + 1 time-delta dim)
        self.lstm      = nn.LSTM(769, 256, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(512, 1)
        # FIX 5: dropout p=0.3 before classification head (Section 3.4.3)
        self.dropout   = nn.Dropout(p=0.3)
        # TWO-LAYER FFN HEAD matching Equation 7: y^ = sigmoid(W2 * ReLU(W1*c + b1) + b2)
        self.fc1       = nn.Linear(512, 256)
        self.fc2       = nn.Linear(256, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, deltas):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state                         # (batch, seq_len, 768)

        # FIX 3: expand deltas (batch, W) -> (batch, seq_len, 1) and concatenate
        batch_size, seq_len, _ = bert_out.shape
        delta_seq = torch.nn.functional.interpolate(
            deltas.unsqueeze(1),                    # (batch, 1, W)
            size=seq_len,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1)                          # (batch, seq_len, 1)

        x            = torch.cat([bert_out, delta_seq], dim=2)  # (batch, seq_len, 769)
        lstm_out, _  = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context      = torch.sum(attn_weights * lstm_out, dim=1)
        # FIX 5: dropout then two-layer FFN head (Equation 7)
        context      = self.dropout(context)
        out          = self.fc2(torch.relu(self.fc1(context)))
        return self.sigmoid(out)

# FIX 5: weights_only=True suppresses PyTorch security warning on newer versions
model = CrisisModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu'),
                                 weights_only=True))
model.eval()

# Input
st.markdown("### Enter a Tweet")
user_input = st.text_area("", height=150)

# Predict
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # ✅ Clean input (VERY IMPORTANT)
        cleaned_text = clean(user_input)

        # Repeat input 10 times to simulate a window — matches training format
        W            = 10
        windowed_text = " ".join([cleaned_text] * W)
        inputs = tokenizer([windowed_text], return_tensors='pt', padding=True,
                           truncation=True, max_length=128)

        inf_deltas = torch.tensor([[t / (W - 1) for t in range(W)]], dtype=torch.float32)

        with torch.no_grad():
            output = model(inputs['input_ids'], inputs['attention_mask'], inf_deltas).view(-1)

        confidence = output.item()

        # FIX 5: improved UI — progress bar + clearer confidence display
        st.markdown("---")
        if confidence > 0.5:
            st.error(f"🚨 **Crisis Detected**")
        else:
            st.success(f"✅ **Not a Crisis**")

        st.markdown(f"**Confidence score:** `{confidence:.3f}`")
        st.progress(float(confidence))

        # Friendly interpretation so the professor can see it at a glance
        if confidence > 0.8:
            st.caption("Very high confidence — strong crisis signal detected.")
        elif confidence > 0.5:
            st.caption("Moderate confidence — possible crisis signal, review recommended.")
        elif confidence > 0.3:
            st.caption("Low confidence — likely not a crisis, but borderline.")
        else:
            st.caption("Very low confidence — not a crisis.")