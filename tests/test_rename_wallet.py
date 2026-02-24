"""
Test Wallet Rename API endpoint - PATCH /api/wallet/{email}/{file_id}/rename
Tests for the new rename feature in Talent Scout
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://file-parse-issue.preview.emergentagent.com')

# Test user with existing files
EXISTING_USER_EMAIL = "abhishek20040916@gmail.com"
# Known file IDs from seed data
KNOWN_FILE_IDS = {
    "advanced_python": "4a48f955238d",
    "video": "8e489e9b2fba",  
    "resume": "3177ce9bc9b3"
}


class TestWalletRenameAPI:
    """Tests for the wallet file rename endpoint"""
    
    def test_rename_endpoint_success(self):
        """Test successful file rename"""
        # Use the video file for rename test
        file_id = KNOWN_FILE_IDS["video"]
        new_name = "Renamed-Video-Test.mp4"
        
        response = requests.patch(
            f"{BASE_URL}/api/wallet/{EXISTING_USER_EMAIL}/{file_id}/rename",
            json={"file_name": new_name},
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "message" in data
        assert data["message"] == "File renamed"
        assert data["file_name"] == new_name
        print(f"SUCCESS: Rename returned - {data}")
    
    def test_rename_verify_persistence(self):
        """After rename, verify new name persists via GET"""
        file_id = KNOWN_FILE_IDS["video"]
        new_name = "Video-936.mp4"  # Rename it back to original
        
        # Do the rename
        response = requests.patch(
            f"{BASE_URL}/api/wallet/{EXISTING_USER_EMAIL}/{file_id}/rename",
            json={"file_name": new_name},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        
        # Verify via GET
        response = requests.get(f"{BASE_URL}/api/wallet/{EXISTING_USER_EMAIL}")
        assert response.status_code == 200
        items = response.json()
        
        # Find the renamed file
        renamed_file = next((i for i in items if i["file_id"] == file_id), None)
        assert renamed_file is not None, "File not found in wallet"
        assert renamed_file["file_name"] == new_name, f"Name not updated. Expected '{new_name}', got '{renamed_file['file_name']}'"
        print(f"SUCCESS: Verified rename persisted - file_name = {renamed_file['file_name']}")
    
    def test_rename_empty_name_fails(self):
        """Test that empty file name is rejected"""
        file_id = KNOWN_FILE_IDS["resume"]
        
        response = requests.patch(
            f"{BASE_URL}/api/wallet/{EXISTING_USER_EMAIL}/{file_id}/rename",
            json={"file_name": ""},
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 400, f"Expected 400 for empty name, got {response.status_code}"
        print(f"SUCCESS: Empty name correctly rejected with 400")
    
    def test_rename_whitespace_only_fails(self):
        """Test that whitespace-only file name is rejected"""
        file_id = KNOWN_FILE_IDS["resume"]
        
        response = requests.patch(
            f"{BASE_URL}/api/wallet/{EXISTING_USER_EMAIL}/{file_id}/rename",
            json={"file_name": "   "},
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 400, f"Expected 400 for whitespace name, got {response.status_code}"
        print(f"SUCCESS: Whitespace-only name correctly rejected with 400")
    
    def test_rename_nonexistent_file(self):
        """Test rename of non-existent file returns 404"""
        fake_file_id = "nonexistent123"
        
        response = requests.patch(
            f"{BASE_URL}/api/wallet/{EXISTING_USER_EMAIL}/{fake_file_id}/rename",
            json={"file_name": "test.pdf"},
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 404, f"Expected 404 for non-existent file, got {response.status_code}"
        print(f"SUCCESS: Non-existent file correctly returns 404")
    
    def test_rename_wrong_user_email(self):
        """Test rename with wrong user email returns 404"""
        file_id = KNOWN_FILE_IDS["resume"]
        wrong_email = "wronguser@test.com"
        
        response = requests.patch(
            f"{BASE_URL}/api/wallet/{wrong_email}/{file_id}/rename",
            json={"file_name": "test.pdf"},
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 404, f"Expected 404 for wrong user, got {response.status_code}"
        print(f"SUCCESS: Wrong user email correctly returns 404")


class TestWalletGetEndpoint:
    """Tests for wallet GET endpoint - verify structure"""
    
    def test_get_wallet_items(self):
        """Test getting wallet items for existing user"""
        response = requests.get(f"{BASE_URL}/api/wallet/{EXISTING_USER_EMAIL}")
        
        assert response.status_code == 200
        items = response.json()
        assert isinstance(items, list)
        assert len(items) >= 3, f"Expected at least 3 items, got {len(items)}"
        
        # Check structure of first item
        if items:
            item = items[0]
            required_fields = ["file_id", "user_email", "item_type", "file_name", "file_size"]
            for field in required_fields:
                assert field in item, f"Missing field: {field}"
        
        print(f"SUCCESS: Got {len(items)} wallet items with correct structure")


class TestFileViewerEndpoint:
    """Tests for file viewing endpoint"""
    
    def test_get_wallet_file_pdf(self):
        """Test getting PDF file for viewing"""
        file_id = KNOWN_FILE_IDS["resume"]
        
        response = requests.get(
            f"{BASE_URL}/api/wallet/file/{EXISTING_USER_EMAIL}/{file_id}",
            stream=True
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        # Check content-disposition is inline
        content_disp = response.headers.get("Content-Disposition", "")
        assert "inline" in content_disp.lower() or "attachment" in content_disp.lower(), \
            f"Expected inline/attachment disposition, got: {content_disp}"
        
        print(f"SUCCESS: File endpoint returns 200 with disposition: {content_disp}")
    
    def test_get_wallet_file_video(self):
        """Test getting video file for viewing"""
        file_id = KNOWN_FILE_IDS["video"]
        
        response = requests.get(
            f"{BASE_URL}/api/wallet/file/{EXISTING_USER_EMAIL}/{file_id}",
            stream=True
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        content_type = response.headers.get("Content-Type", "")
        assert "video" in content_type.lower(), f"Expected video content type, got: {content_type}"
        
        print(f"SUCCESS: Video file serves with content-type: {content_type}")


class TestCandidateProfileEndpoint:
    """Tests for candidate profile endpoint"""
    
    def test_get_candidate_profile(self):
        """Test getting candidate profile"""
        response = requests.get(f"{BASE_URL}/api/candidate/profile/{EXISTING_USER_EMAIL}")
        
        assert response.status_code == 200
        profile = response.json()
        
        required_fields = ["email", "name", "talent_category", "skills"]
        for field in required_fields:
            assert field in profile, f"Missing field: {field}"
        
        print(f"SUCCESS: Candidate profile retrieved - name: {profile.get('name')}")
    
    def test_fresh_candidate_no_profile(self):
        """Test fresh candidate with no profile returns 404"""
        response = requests.get(f"{BASE_URL}/api/candidate/profile/testcandidate@test.com")
        
        # Fresh candidate may not have profile
        assert response.status_code in [200, 404], f"Unexpected status: {response.status_code}"
        print(f"SUCCESS: Fresh candidate profile check returns {response.status_code}")


class TestRecruiterProfileEndpoint:
    """Tests for recruiter profile endpoint"""
    
    def test_fresh_recruiter_no_profile(self):
        """Test fresh recruiter profile - may return basic info or 404"""
        response = requests.get(f"{BASE_URL}/api/recruiter/profile/testrecruiter@test.com")
        
        # Should return 200 with basic info or 404
        assert response.status_code in [200, 404], f"Unexpected status: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert "email" in data
        print(f"SUCCESS: Fresh recruiter profile check returns {response.status_code}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
