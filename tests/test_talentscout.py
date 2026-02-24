"""
Backend tests for Talent Scout API
Tests: root, blockchain eth-status, auth register/verify-otp, recruiter candidates
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://file-parse-issue.preview.emergentagent.com').rstrip('/')

class TestRoot:
    """Root endpoint test"""

    def test_root_returns_version(self):
        response = requests.get(f"{BASE_URL}/api/")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert data.get("message") == "Talent Scout API"
        print(f"Root response: {data}")


class TestBlockchain:
    """Blockchain / Alchemy ETH integration tests"""

    def test_eth_status(self):
        response = requests.get(f"{BASE_URL}/api/blockchain/eth-status", timeout=15)
        assert response.status_code == 200
        data = response.json()
        assert data.get("network") == "eth-mainnet"
        assert "current_block" in data
        assert "alchemy_connected" in data
        print(f"ETH status: {data}")
        assert data["alchemy_connected"] is True, f"Alchemy not connected: {data}"
        assert data["current_block"] != "unknown", "Block number is unknown — Alchemy may be failing"


class TestAuth:
    """Auth endpoint tests"""

    TEST_EMAIL = "test@talentscout.com"
    TEST_NAME = "Test User"
    TEST_ROLE = "candidate"

    def test_register_candidate(self):
        response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": self.TEST_EMAIL,
            "name": self.TEST_NAME,
            "role": self.TEST_ROLE
        })
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data.get("email") == self.TEST_EMAIL
        print(f"Register response: {data}")

    def test_verify_otp_wrong_otp_returns_400(self):
        # First register to ensure user exists
        requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": self.TEST_EMAIL,
            "name": self.TEST_NAME,
            "role": self.TEST_ROLE
        })
        # Now try wrong OTP
        response = requests.post(f"{BASE_URL}/api/auth/verify-otp", json={
            "email": self.TEST_EMAIL,
            "otp": "000000"
        })
        assert response.status_code == 400
        data = response.json()
        print(f"Wrong OTP response: {data}")


class TestRecruiter:
    """Recruiter endpoints"""

    def test_get_candidates_returns_list(self):
        response = requests.get(f"{BASE_URL}/api/recruiter/candidates")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"Candidates count: {len(data)}")
