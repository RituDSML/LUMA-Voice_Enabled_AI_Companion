# --- Evaluate the SAVED model on the held-out TEST set (not validation) ---
# Run this after the training loop and model.save_pretrained() cells above.
# This gives the real, citable test-set metric for Chapter 7 — distinct
# from the per-epoch validation F1 already printed during training.

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

model.eval()
test_loader = DataLoader(test_dataset, batch_size=32)

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['label'].to('cuda')

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
weighted_precision = precision_score(all_labels, all_preds, average='weighted')
weighted_recall = recall_score(all_labels, all_preds, average='weighted')

print(f"TEST SET RESULTS (n={len(all_labels)})")
print(f"Weighted F1:        {weighted_f1:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall:    {weighted_recall:.4f}")
print()
print("Per-class breakdown:")
print(classification_report(all_labels, all_preds, target_names=luma_labels, digits=3))
