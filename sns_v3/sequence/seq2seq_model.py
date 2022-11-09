import transformers
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, EncoderDecoderConfig
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel
import pytorch_lightning as pl
from sns_v3.sequence.load_sequence_dataset import load_sequence_dataset
from sns_v3.sequence.seq2seq_dataset import Seq2SeqDataset, VOCAB


class Seq2SeqModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        encoder_config = BertConfig(max_length=128, vocab_size=256, decoder_start_token_id=1)
        decoder_config = BertConfig(max_length=512, vocab_size=len(VOCAB), decoder_start_token_id=1)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        config.decoder_start_token_id = 1
        config.bos_token_id = 1
        config.pad_token_id = 0
        self.seq2seq = EncoderDecoderModel(config=config)

    def forward(self, X, X_mask):
        return self.seq2seq.generate(input_ids=X, attention_mask=X_mask, max_new_tokens=512)

    def training_step(self, batch, batch_idx):
        X, X_mask, y, y_mask = batch
        loss = self.seq2seq(input_ids=X, attention_mask=X_mask, labels=y).loss
        return loss

    def validation_step(self, batch, batch_idx):
        X, X_mask, y, y_mask = batch
        loss = self.seq2seq(input_ids=X, attention_mask=X_mask, labels=y).loss
        gen = self(X, X_mask)
        # print(gen)
        untok = self.hparams.tokenizer.decode(gen[0].tolist())
        print(untok)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-6)


if __name__ == "__main__":
    ds = Seq2SeqDataset('dataset_100_100', 100)
    test_percent = 0.01
    train_len, test_len = int(len(ds) * (1 - test_percent)), int(len(ds) * test_percent)
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_len, test_len])
    train_dl, test_dl = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True), \
                        DataLoader(test_ds, batch_size=1, shuffle=True, drop_last=True)
    model = Seq2SeqModel(tokenizer=ds.get_tokenizer())
    trainer = pl.Trainer(gpus=1, max_epochs=1000, check_val_every_n_epoch=10)

    trainer.fit(model, train_dl, test_dl)
