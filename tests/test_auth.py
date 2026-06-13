"""Auth: hashing round trips, salting, fail-closed behavior."""
from auth import hash_password, verify_password, check_credentials


def test_round_trip_and_rejections():
    stored = hash_password("S3curePass!")
    assert verify_password("S3curePass!", stored)
    assert not verify_password("S3curePass", stored)
    assert not verify_password("", stored)


def test_unique_salts():
    a, b = hash_password("x"), hash_password("x")
    assert a != b and verify_password("x", a) and verify_password("x", b)


def test_malformed_stored_values_fail_closed():
    for bad in ["nohash", "", None, "$", "salt$"]:
        assert verify_password("x", bad) is False


def test_check_credentials_with_explicit_users():
    users = {"admin": hash_password("admin123")}
    assert check_credentials("admin", "admin123", users)
    assert not check_credentials("admin", "nope", users)
    assert not check_credentials("ghost", "admin123", users)
