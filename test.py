import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./models/cybershield-model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH).to(device)
model.eval()

ID2LABEL = {1: "NON-BULLYING", 0: "BULLYING"}


def predict(text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()

    return ID2LABEL[pred], probs[pred].item()


# =========================
# TEST CASES (INDUSTRY)
# =========================
tests = [
    ("You are stupid", "BULLYING"),
    ("Have a nice day", "NON-BULLYING"),
    ("you are very helpful", "NON-BULLYING"),
    ("tum bewakoof ho", "BULLYING"),
    ("तुम बेवकूफ हो", "BULLYING"),
    ("మీరు చాలా మంచి వాళ్లు", "NON-BULLYING"),
    ("You are a nice guy", "NON-BULLYING"),
    ("I hate you", "BULLYING"),
    ("I love this product", "NON-BULLYING"),
    ("I will Kill you", "BULLYING"),
]

print("\n=== FINAL TEST ===\n")

correct = 0

for text, expected in tests:
    pred, conf = predict(text)

    if pred == expected:
        correct += 1
        status = "✅"
    else:
        status = "❌"

    print(f"{text} → {pred} ({conf:.3f}) | Expected: {expected} {status}")

print("\nAccuracy:", correct / len(tests))
