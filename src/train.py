from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from data_cov_hetero import CovDataset
from data_adk_hetero import AdkDataset
from utils import dotdict
from hetero_net import Hetero_Cryo_Net


def train(config):
    pl.seed_everything(config.seed)

    if config.dataset == "cov":
        dataset = CovDataset(
            root_dir=config.data_dir,
            dose=config.electron_dose,
            prior_idx=config.prior_conf_idx,
        )
    elif config.dataset == "adk":
        dataset = AdkDataset(
            root_dir=config.data_dir,
            dose=config.electron_dose,
            prior_idx=config.prior_conf_idx,
        )
    else:
        raise ValueError("dataset can only be cov or adk")

    model = Hetero_Cryo_Net(
        latent_dim=config.latent_dim,
        dataset=dataset,
        ctf_side_len=config.ctf_side_len,
        output_norm_shift=config.output_norm_shift,
        output_norm_scale=config.output_norm_scale,
        lr=config.lr,
        dcc_loss_factor=config.dcc_loss_factor,
        geom_loss_factor=config.geom_loss_factor,
    )

    # if you want to resume training from a checkpoint:
    # model = Hetero_Cryo_Net.load_from_checkpoint("<ckpt-name>.ckpt", dataset=dataset)

    logger = WandbLogger(
        project="cryoFAT",
        group=config.log_group,
        name=config.name,
        save_dir=config.wandb_dir,
    )

    # log the config of this train run
    logger.experiment.config.update(config)

    # log gradients, parameter histogram and model topology
    logger.watch(model, log="all", log_freq=200)

    trainer = Trainer(
        max_epochs=400,
        logger=logger,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
    )

    train_dl = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=16,
    )
    val_dl = DataLoader(
        dataset.validation_subset(config.val_size), 
        batch_size=config.batch_size,
        num_workers=16,
    )

    # run a full validation loop and log metrics so we know the performance at random init of the model
    trainer.validate(model=model, dataloaders=val_dl)

    # start training
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    config = dotdict()

    config.data_dir = "./../../trainingdata"
    config.wandb_dir = "./../../wandb_logging"
    config.electron_dose = 100
    # CTF of 300x300 has manually been confirmed to have negligible aliasing artefacts
    config.ctf_side_len = 300
    # output shift is equal to the mean of a random set of projections of the prior conf (manually computed with notebook). Scale is determined through trial and error. Note that the network learns additional output scaling itself.
    config.output_norm_shift = -0.11
    config.output_norm_scale = 10.
    config.latent_dim = 8
    config.seed = 1337
    config.batch_size = 128
    config.val_size = 4000
    config.lr = 1e-5
    config.dcc_loss_factor = 0.005
    config.geom_loss_factor = 0.05
    config.prior_conf_idx = 0
    config.dataset = "adk" 
    config.log_group = "viktorprins"
    config.name = f"run_5"

    train(config)
