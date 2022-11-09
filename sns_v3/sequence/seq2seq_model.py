import transformers
from transformers import BertConfig, BertModel
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel
import pytorch_lightning as pl
from sns_v3.sequence.load_sequence_dataset import load_sequence_dataset


class Seq2SeqModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        encoder_config = BertConfig()
        decoder_config = BertConfig()
        encoder = BertGenerationEncoder(encoder_config)
        decoder = BertGenerationDecoder(decoder_config)
        bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    def forward(self, io_samples): 
        print(io_samples)


if __name__ == "__main__":
    model= Seq2SeqModel()
    X, y = load_sequence_dataset('dataset_100_100', 100)
    result = model.forward(X[0])
    print(result)


