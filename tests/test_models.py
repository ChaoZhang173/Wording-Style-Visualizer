"""Offline tests with a real, tiny random BERT (never presented as a trained model)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import BPE, WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import ByteLevel, Whitespace  # noqa: E402
from tokenizers.processors import TemplateProcessing  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast, Qwen2Config, Qwen2Model  # noqa: E402

from token_atlas.models import ModelHandle, _bounded_windows, extract_tokens  # noqa: E402
from token_atlas.types import Article  # noqa: E402


@pytest.fixture
def handle():
    vocab = {token: i for i, token in enumerate(["[UNK]", "[CLS]", "[SEP]", "[PAD]", "a", "b", "c", "d", "e"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", cls_token="[CLS]",
        sep_token="[SEP]", pad_token="[PAD]", model_max_length=12,
    )
    torch.manual_seed(42)
    model = BertModel(BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=24, max_position_embeddings=12,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    ))
    model.eval()
    return ModelHandle(model=model, tokenizer=tokenizer, model_id="offline-random-bert-test", revision="test")


def test_input_lookup_is_identical_for_same_id(handle):
    result = extract_tokens([Article("one", "One", "a b a c a")], handle, mode="input")
    assert result.vectors.dtype == np.float32
    np.testing.assert_array_equal(result.vectors[0], result.vectors[2])
    np.testing.assert_array_equal(result.vectors[0], result.vectors[4])
    expected = handle.model.get_input_embeddings().weight[4].detach().numpy()
    np.testing.assert_array_equal(result.vectors[0], expected)
    assert all(row["token"] not in {"[CLS]", "[SEP]", "[PAD]"} for row in result.tokens)
    assert result.metadata["layer"] is None


def test_contextual_representation_changes_with_context_at_same_position(handle):
    articles = [Article("one", "One", "a b a"), Article("two", "Two", "c b a")]
    result = extract_tokens(articles, handle, mode="contextual", window_size=12)
    # Position 2 has the same token and position in both articles; only context changed.
    assert result.tokens[2]["token_id"] == result.tokens[5]["token_id"]
    assert np.max(np.abs(result.vectors[2] - result.vectors[5])) > 1e-6
    assert result.metadata["resolved_layer"] == 2
    inputs = handle.tokenizer(articles[0].text, return_tensors="pt")
    with torch.inference_mode():
        expected = handle.model(**inputs, output_hidden_states=True).hidden_states[-1][0, 1:-1].numpy()
    np.testing.assert_allclose(result.vectors[:3], expected, atol=1e-6)


def test_overlapping_windows_align_offsets_and_emit_each_occurrence_once(handle):
    text = "a b c d e a b c d e a b c d e"
    result = extract_tokens(
        [Article("one", "One", text)], handle, mode="contextual", window_size=8, overlap=2,
    )
    assert [row["token_index"] for row in result.tokens] == list(range(15))
    assert [row["window_index"] for row in result.tokens] == [0] * 6 + [1] * 4 + [2] * 4 + [3]
    for row in result.tokens:
        assert text[row["char_start"]:row["char_end"]] == row["token"]
        assert row["text"] == text[row["char_start"]:row["char_end"]]
        assert row["excerpt"] == text[max(0, row["char_start"] - 30):row["char_end"] + 30]
    # First new token of window 1 is original token 6, after two overlap tokens.
    with torch.inference_mode():
        window = handle.tokenizer("e a b c d e", return_tensors="pt")
        expected = handle.model(**window, output_hidden_states=True).hidden_states[-1][0, 3].numpy()
    np.testing.assert_allclose(result.vectors[6], expected, atol=1e-6)


def test_limits_are_fair_deterministic_and_report_dropped_tokens(handle):
    articles = [Article(str(i), str(i), "a b c d e a b c d e") for i in range(3)]
    result = extract_tokens(articles, handle, mode="input", max_tokens_per_article=7, max_total_tokens=11)
    assert [entry["retained_tokens"] for entry in result.metadata["article_counts"]] == [4, 4, 3]
    assert result.metadata["original_tokens"] == 30
    assert result.metadata["retained_tokens"] == 11
    assert result.metadata["dropped_tokens"] == 19
    assert len([warning for warning in result.metadata["warnings"] if "未绘制" in warning]) == 3
    for article in articles:
        indices = [row["token_index"] for row in result.tokens if row["article_id"] == article.id]
        assert indices == list(range(len(indices)))


def test_model_window_limit_includes_special_tokens(handle):
    result = extract_tokens([Article("one", "One", "a b c d e " * 4)], handle, window_size=100, overlap=32)
    assert result.metadata["window_size"] == 12
    assert result.metadata["content_window_size"] == 10
    assert result.metadata["overlap"] == 9
    assert len(result.tokens) == 20
    assert result.metadata["warnings"]


def test_invalid_layer_and_duplicate_article_ids_fail_clearly(handle):
    with pytest.raises(ValueError, match="层索引"):
        extract_tokens([Article("one", "One", "a b")], handle, layer=99)
    with pytest.raises(ValueError, match="ID"):
        extract_tokens([Article("same", "One", "a"), Article("same", "Two", "b")], handle)


def test_layer_zero_is_model_initial_state_not_raw_lookup(handle):
    articles = [Article("one", "One", "a b a")]
    raw = extract_tokens(articles, handle, mode="input")
    initial = extract_tokens(articles, handle, mode="contextual", layer=0)
    assert not np.allclose(raw.vectors, initial.vectors)
    assert not np.allclose(initial.vectors[0], initial.vectors[2])
    assert initial.metadata["resolved_layer"] == 0


def test_qwen_decoder_uses_preceding_but_not_future_context(handle):
    torch.manual_seed(7)
    handle.model = Qwen2Model(Qwen2Config(
        vocab_size=9, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=12,
        attention_dropout=0.0,
    ))
    handle.tokenizer.model_input_names = ["input_ids", "attention_mask"]
    result = extract_tokens([
        Article("one", "One", "a b c"),
        Article("two", "Two", "a b d"),
        Article("three", "Three", "c b c"),
    ], handle, window_size=12, overlap=0)
    np.testing.assert_allclose(result.vectors[1], result.vectors[4], atol=1e-6)
    assert np.max(np.abs(result.vectors[1] - result.vectors[7])) > 1e-6


def test_capped_extraction_keeps_exact_full_window_context_without_all_overflows(handle):
    text = "a b c d e " * 20
    article = Article("one", "One", text)
    full = extract_tokens([article], handle, window_size=8, overlap=2)
    capped = extract_tokens([article], handle, window_size=8, overlap=2, max_tokens_per_article=7)
    np.testing.assert_allclose(capped.vectors, full.vectors[:7], atol=1e-6)
    raw = handle.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    windows = _bounded_windows(handle.tokenizer, text, raw, 7, 8, 6, 2)
    # Only a prefix + a small look-ahead is windowed, not all 100 original tokens.
    assert len(windows["input_ids"]) < 6


def test_bounded_windows_preserve_byte_subtokens_with_shared_unicode_offsets():
    alphabet = sorted(ByteLevel.alphabet())
    backend = Tokenizer(BPE(vocab={value: i for i, value in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    text = "雨🌧雨晴 " * 50
    raw = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert len(set(raw["offset_mapping"])) < len(raw["offset_mapping"])
    windows = _bounded_windows(tokenizer, text, raw, 17, 8, 8, 2)
    for index in range(3):
        assert windows["input_ids"][index] == raw["input_ids"][index * 6:index * 6 + 8]
        assert windows["offset_mapping"][index] == raw["offset_mapping"][index * 6:index * 6 + 8]
    assert len(windows["input_ids"]) < 8
