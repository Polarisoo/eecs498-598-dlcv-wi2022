import pathlib
import sys

import torch
from torch import nn

A5_DIR = pathlib.Path(__file__).resolve().parents[1] / "assignments" / "a5-rnn-transformers"
sys.path.insert(0, str(A5_DIR))

import rnn_lstm_captioning as cap
import transformers as tr


class FakeImageEncoder(nn.Module):
    def __init__(self, pretrained=True, verbose=True):
        super().__init__()
        self._out_channels = 4

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, images):
        n = images.shape[0]
        vals = torch.linspace(0.0, 1.0, steps=n * 4 * 4 * 4, device=images.device, dtype=images.dtype)
        return vals.reshape(n, 4, 4, 4)


def test_rnn_step_and_sequence_backward_match_autograd():
    torch.manual_seed(0)
    N, T, D, H = 2, 3, 4, 5
    x = torch.randn(N, T, D, dtype=torch.double, requires_grad=True)
    h0 = torch.randn(N, H, dtype=torch.double, requires_grad=True)
    Wx = torch.randn(D, H, dtype=torch.double, requires_grad=True)
    Wh = torch.randn(H, H, dtype=torch.double, requires_grad=True)
    b = torch.randn(H, dtype=torch.double, requires_grad=True)

    h, cache = cap.rnn_forward(x, h0, Wx, Wh, b)
    dh = torch.randn_like(h)
    dx, dh0, dWx, dWh, db = cap.rnn_backward(dh, cache)
    h.backward(dh)

    assert torch.allclose(dx, x.grad, atol=1e-9)
    assert torch.allclose(dh0, h0.grad, atol=1e-9)
    assert torch.allclose(dWx, Wx.grad, atol=1e-9)
    assert torch.allclose(dWh, Wh.grad, atol=1e-9)
    assert torch.allclose(db, b.grad, atol=1e-9)


def test_embeddings_temporal_loss_lstm_and_attention_shapes():
    torch.manual_seed(0)
    emb = cap.WordEmbedding(vocab_size=7, embed_size=6)
    out = emb(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert out.shape == (2, 3, 6)
    assert torch.allclose(out[0, 0], emb.W_embed[1])

    scores = torch.randn(2, 3, 7)
    y = torch.tensor([[1, 2, 0], [3, 0, 4]])
    loss = cap.temporal_softmax_loss(scores, y, ignore_index=0)
    expected = torch.nn.functional.cross_entropy(scores.permute(0, 2, 1), y, ignore_index=0, reduction="sum") / 2
    assert torch.allclose(loss, expected)

    lstm = cap.LSTM(input_dim=6, hidden_dim=5)
    hidden = lstm(torch.randn(2, 4, 6), torch.randn(2, 5))
    assert hidden.shape == (2, 4, 5)

    A = torch.randn(2, 5, 4, 4)
    attn, weights = cap.dot_product_attention(torch.randn(2, 5), A)
    assert attn.shape == (2, 5)
    assert weights.shape == (2, 4, 4)
    assert torch.allclose(weights.sum(dim=(1, 2)), torch.ones(2), atol=1e-6)

    attn_lstm = cap.AttentionLSTM(input_dim=6, hidden_dim=5)
    assert attn_lstm(torch.randn(2, 4, 6), A).shape == (2, 4, 5)


def test_captioning_rnn_forward_and_sample_with_fake_encoder(monkeypatch):
    monkeypatch.setattr(cap, "ImageEncoder", FakeImageEncoder)
    word_to_idx = {"<NULL>": 0, "<START>": 1, "<END>": 2, "cat": 3, "dog": 4}
    images = torch.zeros(2, 3, 32, 32)
    captions = torch.tensor([[1, 3, 4, 2, 0], [1, 4, 3, 2, 0]])

    for cell_type in ["rnn", "lstm", "attn"]:
        model = cap.CaptioningRNN(
            word_to_idx,
            input_dim=4,
            wordvec_dim=6,
            hidden_dim=5,
            cell_type=cell_type,
            image_encoder_pretrained=False,
            ignore_index=0,
        )
        loss = model(images, captions)
        assert torch.isfinite(loss)
        sampled = model.sample(images, max_length=4)
        if cell_type == "attn":
            sampled, attn_weights = sampled
            assert attn_weights.shape == (2, 4, 4, 4)
        assert sampled.shape == (2, 4)
        assert sampled.dtype == torch.long


def test_transformer_attention_blocks_and_forward():
    torch.manual_seed(0)
    vocab = ["BOS", "EOS", "POSITIVE", "NEGATIVE", "add", "subtract", "0", "1", "2", "3"]
    token_dict = tr.generate_token_dict(vocab)
    assert token_dict["BOS"] == 0
    assert tr.prepocess_input_sequence("BOS POSITIVE 012 add NEGATIVE 003 EOS", token_dict, vocab[:6]) == [0, 2, 6, 7, 8, 4, 3, 6, 6, 9, 1]

    q = torch.randn(2, 4, 8, requires_grad=True)
    single = tr.scaled_dot_product_two_loop_single(q[0], q[0], q[0])
    batched = tr.scaled_dot_product_two_loop_batch(q, q, q)
    no_loop, weights = tr.scaled_dot_product_no_loop_batch(q, q, q)
    assert torch.allclose(single, no_loop[0], atol=1e-6)
    assert torch.allclose(batched, no_loop, atol=1e-6)
    assert weights.shape == (2, 4, 4)

    mask = tr.get_subsequent_mask(torch.zeros(2, 4, dtype=torch.long))
    assert mask.shape == (2, 4, 4)
    assert mask[0, 0, 1] and not mask[0, 1, 0]

    assert tr.SelfAttention(8, 4, 4)(q, q, q, mask).shape == (2, 4, 4)
    assert tr.MultiHeadAttention(2, 8, 4)(q, q, q, mask).shape == (2, 4, 8)
    assert tr.LayerNormalization(8)(q).shape == (2, 4, 8)
    assert tr.FeedForwardBlock(8, 16)(q).shape == (2, 4, 8)
    enc = tr.EncoderBlock(2, 8, 16, 0.0)
    assert hasattr(enc, "MultiHeadBlock")
    assert enc(q).shape == (2, 4, 8)
    assert tr.DecoderBlock(2, 8, 16, 0.0)(q, q, mask).shape == (2, 4, 8)

    simple_pos = tr.position_encoding_simple(4, 8)
    sinus_pos = tr.position_encoding_sinusoid(4, 8)
    assert simple_pos.shape == (1, 4, 8)
    assert sinus_pos.shape == (1, 4, 8)

    model = tr.Transformer(2, 8, 16, 0.0, 1, 1, len(vocab))
    ques = torch.randint(0, len(vocab), (2, 5))
    ans = torch.randint(0, len(vocab), (2, 6))
    out = model(
        ques,
        tr.position_encoding_simple(5, 8).repeat(2, 1, 1),
        ans,
        tr.position_encoding_simple(6, 8).repeat(2, 1, 1),
    )
    assert out.shape == (2 * 5, len(vocab))
    out.sum().backward()
