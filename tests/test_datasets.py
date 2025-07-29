import socket
import urllib.error

import pytest

from wefe.datasets.datasets import (
    _retry_request,
    fetch_debias_multiclass,
    fetch_debiaswe,
    fetch_eds,
    fetch_gn_glove,
    load_bingliu,
    load_weat,
)


def test_load_bingliu() -> None:
    bingliu = load_bingliu()
    assert isinstance(bingliu, dict)
    assert list(bingliu.keys()) == ["positive_words", "negative_words"]
    assert len(list(bingliu.keys())) == 2

    for set_name, set_ in bingliu.items():
        assert isinstance(set_name, str)
        assert isinstance(set_, list)
        assert len(set_) > 0
        for word in set_:
            assert isinstance(word, str)
            assert len(word) > 0


def test_fetch_eds() -> None:
    eds_dataset = fetch_eds()
    assert isinstance(eds_dataset, dict)
    assert list(eds_dataset.keys()) == [
        "adjectives_appearance",
        "adjectives_otherization",
        "adjectives_sensitive",
        "names_asian",
        "names_black",
        "names_chinese",
        "names_hispanic",
        "names_russian",
        "names_white",
        "words_christianity",
        "words_islam",
        "words_terrorism",
        "male_occupations",
        "female_occupations",
        "occupations_white",
        "occupations_black",
        "occupations_asian",
        "occupations_hispanic",
        "male_terms",
        "female_terms",
        "adjectives_intelligence",
    ]
    assert len(list(eds_dataset.keys())) == 21

    for set_name, set_ in eds_dataset.items():
        assert isinstance(set_name, str)
        assert isinstance(set_, list)
        assert len(set_) > 0
        for word in set_:
            assert isinstance(word, str)
            assert len(word) > 0


def test_fetch_debiaswe() -> None:
    debiaswe_datatset = fetch_debiaswe()
    assert isinstance(debiaswe_datatset, dict)
    assert list(debiaswe_datatset.keys()) == [
        "male_terms",
        "female_terms",
        "definitional_pairs",
        "equalize_pairs",
        "gender_specific",
        "professions",
    ]
    assert len(list(debiaswe_datatset.keys())) == 6

    for set_name, set_ in debiaswe_datatset.items():
        assert isinstance(set_name, str)
        assert isinstance(set_, list)
        assert len(set_) > 0
        for word in set_:
            assert isinstance(word, (str, list))
            assert len(word) > 0


def test_fetch_debias_multiclass() -> None:
    debias_multiclass_dataset = fetch_debias_multiclass()
    assert isinstance(debias_multiclass_dataset, dict)
    assert list(debias_multiclass_dataset.keys()) == [
        "male_terms",
        "female_terms",
        "male_roles",
        "female_roles",
        "black_terms",
        "white_terms",
        "asian_terms",
        "black_biased_words",
        "white_biased_words",
        "asian_biased_words",
        "judaism_terms",
        "christianity_terms",
        "islam_terms",
        "greed",
        "conservative",
        "terrorism",
        "gender_definitional_sets",
        "ethnicity_definitional_sets",
        "religion_definitional_sets",
        "gender_analogy_templates",
        "ethnicity_analogy_templates",
        "religion_analogy_templates",
        "gender_eval_target",
        "ethnicity_eval_target",
        "religion_eval_target",
    ]

    assert len(list(debias_multiclass_dataset.keys())) == 25

    for set_name, set_ in debias_multiclass_dataset.items():
        assert isinstance(set_name, str)
        assert isinstance(set_, (list, dict))
        if isinstance(set_, list):
            assert len(set_) > 0
            for word in set_:
                assert isinstance(word, (str, list))
                assert len(word) > 0


def test_load_weat() -> None:
    weat = load_weat()
    assert isinstance(weat, dict)
    assert list(weat.keys()) == [
        "flowers",
        "insects",
        "pleasant_5",
        "unpleasant_5a",
        "instruments",
        "weapons",
        "european_american_names_5",
        "african_american_names_5",
        "unpleasant_5b",
        "european_american_names_7",
        "african_american_names_7",
        "pleasant_9",
        "unpleasant_9",
        "male_names",
        "female_names",
        "career",
        "family",
        "math",
        "arts",
        "male_terms",
        "female_terms",
        "science",
        "arts_2",
        "male_terms_2",
        "female_terms_2",
        "mental_disease",
        "physical_disease",
        "temporary",
        "permanent",
        "young_people_names",
        "old_people_names",
    ]
    for set_name, set_ in weat.items():
        assert isinstance(set_name, str)
        assert isinstance(set_, list)
        assert len(set_) > 0
        for word in set_:
            assert isinstance(word, str)
            assert len(word) > 0


def test_load_gn_glove() -> None:
    gn_glove_words = fetch_gn_glove()
    assert isinstance(gn_glove_words, dict)
    assert list(gn_glove_words.keys()) == ["male_terms", "female_terms"]
    for set_name, set_ in gn_glove_words.items():
        assert isinstance(set_name, str)
        assert isinstance(set_, list)
        assert len(set_) > 0
        for word in set_:
            assert isinstance(word, str)
            assert len(word) > 0


# Tests for retry functionality
class TestRetryRequest:
    """Test cases for the _retry_request function."""

    def test_retry_request_success_on_first_attempt(self):
        """Test _retry_request result when function succeeds on first attempt."""
        from unittest.mock import Mock

        mock_func = Mock(return_value="success")

        result = _retry_request(mock_func, "arg1", "arg2", kwarg1="value1")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")

    def test_retry_request_rate_limit_error(self, monkeypatch):
        """Test retry behavior for HTTP 429 rate limit errors."""
        from unittest.mock import Mock

        mock_sleep = Mock()
        mock_warning = Mock()
        monkeypatch.setattr("time.sleep", mock_sleep)
        monkeypatch.setattr("logging.warning", mock_warning)

        mock_func = Mock()

        # Create HTTPError with code 429
        from email.message import EmailMessage

        headers = EmailMessage()
        http_error = urllib.error.HTTPError(
            url="http://test.com",
            code=429,
            msg="Too Many Requests",
            hdrs=headers,
            fp=None,
        )

        # First two calls fail with 429, third succeeds
        mock_func.side_effect = [http_error, http_error, "success"]

        result = _retry_request(mock_func, n_retries=3)

        assert result == "success"
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2
        assert mock_warning.call_count == 2

        # Check exponential backoff sleep times
        mock_sleep.assert_any_call(1)  # 2^0 = 1
        mock_sleep.assert_any_call(2)  # 2^1 = 2

    def test_retry_request_timeout_error(self, monkeypatch):
        """Test retry behavior for timeout errors."""
        from unittest.mock import Mock

        mock_sleep = Mock()
        mock_warning = Mock()
        monkeypatch.setattr("time.sleep", mock_sleep)
        monkeypatch.setattr("logging.warning", mock_warning)

        mock_func = Mock()

        # First call fails with timeout, second succeeds
        mock_func.side_effect = [socket.timeout("Connection timeout"), "success"]

        result = _retry_request(mock_func, n_retries=2)

        assert result == "success"
        assert mock_func.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1
        mock_warning.assert_called_once()

    def test_retry_request_timeout_error_os_error(self, monkeypatch):
        """Test retry behavior for OSError (network timeout)."""
        from unittest.mock import Mock

        mock_sleep = Mock()
        mock_warning = Mock()
        monkeypatch.setattr("time.sleep", mock_sleep)
        monkeypatch.setattr("logging.warning", mock_warning)

        mock_func = Mock()

        # First call fails with OSError, second succeeds
        mock_func.side_effect = [OSError("Network timeout"), "success"]

        result = _retry_request(mock_func, n_retries=2)

        assert result == "success"
        assert mock_func.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1
        mock_warning.assert_called_once()

    def test_retry_request_generic_exception(self, monkeypatch):
        """Test retry behavior for generic exceptions."""
        from unittest.mock import Mock

        mock_sleep = Mock()
        mock_warning = Mock()
        monkeypatch.setattr("time.sleep", mock_sleep)
        monkeypatch.setattr("logging.warning", mock_warning)

        mock_func = Mock()

        # First call fails with generic exception, second succeeds
        mock_func.side_effect = [ValueError("Generic error"), "success"]

        result = _retry_request(mock_func, n_retries=2)

        assert result == "success"
        assert mock_func.call_count == 2
        mock_sleep.assert_called_once_with(1)  # Fixed 1-second delay
        mock_warning.assert_called_once()

    def test_retry_request_non_retryable_http_error(self):
        """Test that non-retryable HTTP errors are not retried."""
        from unittest.mock import Mock

        mock_func = Mock()

        # 404 Not Found should not be retried
        from email.message import EmailMessage

        headers = EmailMessage()
        http_error = urllib.error.HTTPError(
            url="http://test.com", code=404, msg="Not Found", hdrs=headers, fp=None
        )
        mock_func.side_effect = http_error

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _retry_request(mock_func, n_retries=3)

        assert exc_info.value.code == 404
        mock_func.assert_called_once()  # Should only be called once

    def test_retry_request_exhaust_retries(self, monkeypatch):
        """Test that function raises exception when all retries are exhausted."""
        from unittest.mock import Mock

        mock_sleep = Mock()
        mock_warning = Mock()
        monkeypatch.setattr("time.sleep", mock_sleep)
        monkeypatch.setattr("logging.warning", mock_warning)

        mock_func = Mock()

        # Always fail with rate limit error
        from email.message import EmailMessage

        headers = EmailMessage()
        http_error = urllib.error.HTTPError(
            url="http://test.com",
            code=429,
            msg="Too Many Requests",
            hdrs=headers,
            fp=None,
        )
        mock_func.side_effect = http_error

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _retry_request(mock_func, n_retries=2)

        assert exc_info.value.code == 429
        assert mock_func.call_count == 3  # Initial call + 2 retries
        assert mock_sleep.call_count == 2
        assert mock_warning.call_count == 2

    def test_retry_request_url_error(self):
        """Test that URLError without code is not retried."""
        from unittest.mock import Mock

        mock_func = Mock()

        url_error = urllib.error.URLError("Connection failed")
        mock_func.side_effect = url_error

        with pytest.raises(urllib.error.URLError):
            _retry_request(mock_func, n_retries=3)

        mock_func.assert_called_once()  # Should only be called once
