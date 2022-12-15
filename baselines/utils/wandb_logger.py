import wandb
import matplotlib.pyplot as plt

class WandbLogger():
    def __init__(self, args):
        self.args = args
        self.run = wandb.init(reinit=True,
                                project=args.wandb_logging.wandb_project,
                                config=args
                                )

    def log(self, dct):
        wandb.log(dct)

    def log_plots(self, gnd, pred, names):
        assert len(gnd) == 6
        fig, ax = plt.subplots(2, 3, figsize=(12, 4))
        for j in range(3):
            ax[0][j].imshow(gnd[j])
            ax[1][j].imshow(pred[j].T)
            ax[0][j].set_title(names[j])
        plt.tight_layout()
        wandb.log({"mels1": wandb.Image(plt)})
        plt.clf()
        fig, ax = plt.subplots(2, 3, figsize=(12, 4))
        for j in range(3):
            ax[0][j].imshow(gnd[j+3])
            ax[1][j].imshow(pred[j+3].T)
            ax[0][j].set_title(names[j+3])
        plt.tight_layout()
        wandb.log({"mels2": wandb.Image(plt)})
    
    def summary(self, dct):
        for key in dct:
            wandb.run.summary[key] = dct[key]

    def end_run(self):
        self.run.finish()

    def log_audio(self, aud, names):
        upload_dict = {names[i]:wandb.Audio(aud[i],  sample_rate=self.args.signal.sampling_rate) for i in range(len(aud))}
        wandb.log(upload_dict)
