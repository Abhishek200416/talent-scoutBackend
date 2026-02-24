# Password Authentication Endpoints for server_sqlite.py
# Add these after the existing auth endpoints

# ---- Password Authentication ----
@api_router.post("/auth/set-password")
async def set_password(request: SetPasswordRequest):
    """Set or update password for a user"""
    async with async_session() as s:
        result = await s.execute(select(User).where(User.email == request.email))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if len(request.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        
        password_hash = hash_password(request.password)
        user.password_hash = password_hash
        user.password_set_at = _now()
        await s.commit()
    
    return {"message": "Password set successfully"}

@api_router.post("/auth/login-password")
async def login_with_password(request: PasswordLoginRequest):
    """Login using password instead of OTP"""
    async with async_session() as s:
        result = await s.execute(select(User).where(User.email == request.email))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user.verified:
            raise HTTPException(status_code=400, detail="Email not verified. Please use OTP login first.")
        
        if not user.password_hash:
            raise HTTPException(status_code=400, detail="No password set. Please use OTP login or set a password first.")
        
        if not verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=400, detail="Invalid password")
    
    return {
        "message": "Login successful",
        "token": "token_" + request.email,
        "role": user.role,
        "name": user.name,
        "email": request.email
    }

@api_router.post("/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    """Request OTP for password reset"""
    async with async_session() as s:
        result = await s.execute(select(User).where(User.email == request.email))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        otp = _otp()
        otp_expiry = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        
        user.password_reset_otp = otp
        user.password_reset_otp_expiry = otp_expiry
        user.last_otp_sent = _now()
        await s.commit()
        
        name = user.name or "there"
        background_tasks.add_task(send_email, request.email, "Talent Scout - Password Reset Code",
            f"""<h2 style="color: #6366f1;">Hello {name}!</h2>
            <p>You requested to reset your password. Here's your verification code:</p>
            <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 30px; text-align: center; border-radius: 10px; margin: 20px 0;">
                <h1 style="color: white; font-size: 48px; letter-spacing: 8px; margin: 0;">{otp}</h1>
            </div>
            <p style="color: #666;">This code expires in <strong>10 minutes</strong>.</p>
            <p style="color: #999; font-size: 12px;">If you did not request this, please ignore this email and your password will remain unchanged.</p>""")
    
    return {"message": "Password reset OTP sent to email"}

@api_router.post("/auth/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """Reset password using OTP"""
    async with async_session() as s:
        result = await s.execute(select(User).where(User.email == request.email))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user.password_reset_otp:
            raise HTTPException(status_code=400, detail="Please request a password reset first")
        
        if user.password_reset_otp != request.otp:
            raise HTTPException(status_code=400, detail="Invalid OTP")
        
        if user.password_reset_otp_expiry:
            otp_expiry = datetime.fromisoformat(user.password_reset_otp_expiry)
            if datetime.now(timezone.utc) > otp_expiry:
                raise HTTPException(status_code=400, detail="OTP expired. Please request a new one.")
        
        if len(request.new_password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        
        password_hash = hash_password(request.new_password)
        user.password_hash = password_hash
        user.password_reset_otp = None
        user.password_reset_otp_expiry = None
        user.password_set_at = _now()
        await s.commit()
    
    return {"message": "Password reset successfully"}

@api_router.get("/auth/check-password/{email}")
async def check_password_set(email: str):
    """Check if user has a password set"""
    async with async_session() as s:
        result = await s.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
    
    return {"has_password": bool(user.password_hash)}

# ---- AI Confirm Skills ----
@api_router.post("/ai/confirm-skills/{email}")
async def confirm_skills(email: str, request: dict):
    """Add confirmed skills to user profile"""
    skills_to_add = request.get("skills", [])
    if not skills_to_add:
        return {"message": "No skills to add"}
    
    async with async_session() as s:
        result = await s.execute(select(CandidateProfile).where(CandidateProfile.email == email))
        profile = result.scalar_one_or_none()
        
        if profile:
            existing_skills = json.loads(profile.skills or "[]")
            all_skills = existing_skills + skills_to_add
            unique_skills = []
            seen_lower = set()
            for skill in all_skills:
                skill_lower = skill.lower()
                if skill_lower not in seen_lower:
                    unique_skills.append(skill)
                    seen_lower.add(skill_lower)
            
            profile.skills = json.dumps(unique_skills)
            profile.updated_at = _now()
            await s.commit()
            
            return {"message": f"Added {len(skills_to_add)} skills to profile", "total_skills": len(unique_skills)}
    
    return {"message": "Profile not found"}

# ---- AI Parse Job Document ----
@api_router.post("/ai/parse-job-document/{email}")
async def parse_job_document(email: str, file: UploadFile = File(...)):
    """Parse job description from uploaded document"""
    if not AI_KEY:
        raise HTTPException(status_code=500, detail="AI service not configured")
    
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB")
    
    temp_file = Path(f"/tmp/job_doc_{uuid.uuid4().hex}.{file.filename.split('.')[-1]}")
    temp_file.write_bytes(contents)
    
    try:
        mime_type = file.content_type or "application/pdf"
        extracted_text = ""
        
        if "word" in mime_type.lower() or file.filename.endswith(('.doc', '.docx')):
            try:
                import docx
                doc = docx.Document(str(temp_file))
                extracted_text = "\n".join([para.text for para in doc.paragraphs])
            except:
                pass
        
        chat = LlmChat(
            api_key=AI_KEY,
            session_id=f"job_parse_{uuid.uuid4().hex[:8]}",
            system_message="You are an expert HR AI. Extract job posting details from documents. Always respond with valid JSON only."
        ).with_model("gemini", "gemini-2.5-flash")
        
        if extracted_text:
            prompt = f"""Extract job posting details from this text. Return ONLY valid JSON:
{{
  "title": "Job Title",
  "company_name": "Company Name",
  "location": "City, State/Country",
  "job_type": "Full-time|Part-time|Contract|Internship",
  "salary_range": "Salary range if mentioned",
  "talent_category": "Technical (IT/Engineering)|Business/Marketing|Creative/Design|Healthcare|Finance|Education|Other",
  "required_skills": ["skill1", "skill2", "skill3"],
  "description": "Full job description text",
  "responsibilities": ["resp1", "resp2"],
  "qualifications": ["qual1", "qual2"],
  "experience_required": "2+ years|Entry Level|etc"
}}

Job text:
{extracted_text[:8000]}"""
            response = await chat.send_message(UserMessage(text=prompt))
        else:
            file_content = FileContentWithMimeType(
                file_path=str(temp_file),
                mime_type=mime_type
            )
            
            prompt = """Extract job posting details from this document/image. Return ONLY valid JSON:
{
  "title": "Job Title",
  "company_name": "Company Name",
  "location": "City, State/Country",
  "job_type": "Full-time|Part-time|Contract|Internship",
  "salary_range": "Salary range if mentioned",
  "talent_category": "Technical (IT/Engineering)|Business/Marketing|Creative/Design|Healthcare|Finance|Education|Other",
  "required_skills": ["skill1", "skill2", "skill3"],
  "description": "Full job description text",
  "responsibilities": ["resp1", "resp2"],
  "qualifications": ["qual1", "qual2"],
  "experience_required": "2+ years|Entry Level|etc"
}

Extract all job details from this document."""
            
            response = await chat.send_message(UserMessage(text=prompt, file_contents=[file_content]))
        
        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            job_data = json.loads(clean)
        except json.JSONDecodeError:
            job_data = {"error": "Could not parse job details", "raw_response": response}
        
        return {"message": "Job document parsed successfully", "job_data": job_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse job document: {str(e)}")
    finally:
        if temp_file.exists():
            temp_file.unlink()
