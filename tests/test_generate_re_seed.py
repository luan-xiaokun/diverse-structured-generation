from unittest.mock import Mock

from scripts.generate_re import GeneratedData, set_generation_seed


def test_generated_data_round_trips_seed():
    data = GeneratedData(
        grammar="css-color",
        regex="#[0-9a-fA-F]{6}",
        prompt="Give me a CSS color code.",
        model="test/model",
        max_tokens=18,
        top_k=None,
        top_p=None,
        temperature=None,
        seed=42,
        samples=["#aabbcc"],
    )

    payload = data.to_dict()
    restored = GeneratedData.from_dict(payload)

    assert payload["seed"] == 42
    assert restored.seed == 42


def test_generated_data_accepts_legacy_payload_without_seed():
    restored = GeneratedData.from_dict(
        {
            "grammar": "css-color",
            "regex": "#[0-9a-fA-F]{6}",
            "prompt": "Give me a CSS color code.",
            "model": "test/model",
            "max_tokens": 18,
            "top_k": None,
            "top_p": None,
            "temperature": None,
            "samples": ["#aabbcc"],
        }
    )

    assert restored.seed is None


def test_set_generation_seed_seeds_python_numpy_and_torch(monkeypatch):
    random_seed = Mock()
    numpy_seed = Mock()
    torch_manual_seed = Mock()
    torch_cuda_manual_seed_all = Mock()

    monkeypatch.setattr("scripts.generate_re.random.seed", random_seed)
    monkeypatch.setattr("scripts.generate_re.np.random.seed", numpy_seed)
    monkeypatch.setattr("scripts.generate_re.torch.manual_seed", torch_manual_seed)
    monkeypatch.setattr(
        "scripts.generate_re.torch.cuda.is_available", lambda: True
    )
    monkeypatch.setattr(
        "scripts.generate_re.torch.cuda.manual_seed_all",
        torch_cuda_manual_seed_all,
    )

    set_generation_seed(42)

    random_seed.assert_called_once_with(42)
    numpy_seed.assert_called_once_with(42)
    torch_manual_seed.assert_called_once_with(42)
    torch_cuda_manual_seed_all.assert_called_once_with(42)


def test_set_generation_seed_noops_when_seed_is_none(monkeypatch):
    random_seed = Mock()
    numpy_seed = Mock()
    torch_manual_seed = Mock()

    monkeypatch.setattr("scripts.generate_re.random.seed", random_seed)
    monkeypatch.setattr("scripts.generate_re.np.random.seed", numpy_seed)
    monkeypatch.setattr("scripts.generate_re.torch.manual_seed", torch_manual_seed)

    set_generation_seed(None)

    random_seed.assert_not_called()
    numpy_seed.assert_not_called()
    torch_manual_seed.assert_not_called()
