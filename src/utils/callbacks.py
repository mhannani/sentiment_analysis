from transformers import TrainerCallback

class PrintTrainLossCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model, **kwargs):
        train_loss = state
        print(f"Training loss: {train_loss}")