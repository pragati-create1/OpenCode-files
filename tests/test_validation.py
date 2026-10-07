import pytest
from app.utils.validation import validate_upload, detect_media_type, safe_filename, ValidationError


def test_safe_filename():
    assert safe_filename("my file!.jpg") == "my_file_.jpg"


def test_detect_media_type(sample_image, sample_wav):
    assert detect_media_type(sample_image) == "image"
    assert detect_media_type(sample_wav) == "audio"
    with pytest.raises(ValidationError):
        detect_media_type("x.txt")


def test_validate_upload(sample_image):
    assert validate_upload(sample_image, 10_000_000) == "image"
    with pytest.raises(ValidationError):
        validate_upload(sample_image, 10)  # too small limit
