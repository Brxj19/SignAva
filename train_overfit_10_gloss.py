from src.model.train import train_model

history = train_model(
    epochs=1000,
    batch_size=4,
    learning_rate=1e-4,
    dropout=0.0,
    loss_type="weighted",
    sequence_mode="resample",
    resume_from=None,
    save_every=50
)

print("10-gloss overfit training complete.")
