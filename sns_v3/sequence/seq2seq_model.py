import transformers
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, EncoderDecoderConfig
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel
import pytorch_lightning as pl
from sns_v3.sequence.load_sequence_dataset import load_sequence_dataset
from sns_v3.sequence.seq2seq_dataset import Seq2SeqDataset, VOCAB
import json


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
        self.untoks = {}

    def forward(self, **kwargs):
        return self.seq2seq(**kwargs)

    def generate(self, **kwargs):
        return self.seq2seq.generate(**kwargs)

    def training_step(self, batch, batch_idx):
        X, X_mask, y, y_mask = batch
        loss = self.seq2seq(input_ids=X, attention_mask=X_mask, labels=y).loss
        return loss
    
    def generate_callback(self, X, X_mask, batch_idx):
        gen = self.generate(input_ids=X,
                            attention_mask=X_mask,
                            max_length=512,
                            do_sample=True,
                            num_return_sequences=8)
        gen_untoks = []
        print(f"Sampled {gen.shape[0]} sequences")
        for i in range(gen.shape[0]):
            decoded = self.hparams.tokenizer.decode(gen[i])
            gen_untoks.append(decoded)

        if self.global_step not in self.untoks.keys():
            self.untoks[self.global_step] = {}
        self.untoks[self.global_step][batch_idx] = {}
        self.untoks[self.global_step][batch_idx]['gen'] = gen_untoks
        self.untoks[self.global_step][batch_idx]['X'] = X.detach().cpu().tolist()

        json_f = open("seq2seq.json", "w")
        json.dump(self.untoks, json_f)
        json_f.close()

    def validation_step(self, batch, batch_idx):
        X, X_mask, y, y_mask = batch
        loss = self.seq2seq(input_ids=X, attention_mask=X_mask, labels=y).loss
        self.generate_callback(X, X_mask, batch_idx)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)


if __name__ == "__main__":
    ds = Seq2SeqDataset('dataset_50_100', 100000)
    test_percent = 0.0001
    train_len, test_len = int(len(ds) * (1 - test_percent)), int(len(ds) * test_percent)
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_len, test_len])
    train_dl, test_dl = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True), \
                        DataLoader(test_ds, batch_size=1, shuffle=True, drop_last=True)
    model = Seq2SeqModel(tokenizer=ds.get_tokenizer())
    trainer = pl.Trainer(gpus=1, max_epochs=1000, val_check_interval=0.01)

    trainer.fit(model, train_dl, test_dl)
