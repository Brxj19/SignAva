from src.model.train import train_model

history = train_model(
    epochs=300,
    batch_size=8,
    learning_rate=3e-5,
    dropout=0.05,
    loss_type="weighted",
    sequence_mode="resample",
    resume_from=None,
    save_every=25
)

print("Training complete.")
