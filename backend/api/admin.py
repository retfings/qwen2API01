from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from backend.core.config import settings
from backend.core.database import AsyncJsonDB
from backend.core.account_pool import AccountPool, Account

router = APIRouter()

def verify_admin(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split("Bearer ")[1]
    if token != settings.ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Admin Key Mismatch")
    return token

class UserCreate(BaseModel):
    name: str
    quota: int = 1000000

class User(BaseModel):
    id: str
    name: str
    quota: int
    used_tokens: int

@router.get("/status", dependencies=[Depends(verify_admin)])
async def get_system_status(request):
    # 这里要接入全局引擎的状态
    pool = request.app.state.account_pool
    engine = request.app.state.browser_engine
    return {
        "accounts": pool.status(),
        "browser_engine": {
            "started": engine._started,
            "pool_size": engine.pool_size,
            "queue": engine._pages.qsize()
        }
    }

@router.get("/users", dependencies=[Depends(verify_admin)])
async def list_users(request):
    db: AsyncJsonDB = request.app.state.users_db
    data = await db.get()
    return {"users": data}

@router.post("/users", dependencies=[Depends(verify_admin)])
async def create_user(user: UserCreate, request):
    import uuid
    db: AsyncJsonDB = request.app.state.users_db
    data = await db.get()
    new_user = {
        "id": f"sk-{uuid.uuid4().hex}",
        "name": user.name,
        "quota": user.quota,
        "used_tokens": 0
    }
    data.append(new_user)
    await db.save(data)
    return new_user

@router.get("/accounts", dependencies=[Depends(verify_admin)])
async def list_accounts(request):
    pool: AccountPool = request.app.state.account_pool
    return {"accounts": [a.to_dict() for a in pool.accounts]}

@router.post("/accounts", dependencies=[Depends(verify_admin)])
async def add_account(acc: dict, request):
    pool: AccountPool = request.app.state.account_pool
    await pool.add(Account(**acc))
    return {"status": "success"}

@router.delete("/accounts/{email}", dependencies=[Depends(verify_admin)])
async def delete_account(email: str, request):
    pool: AccountPool = request.app.state.account_pool
    await pool.remove(email)
    return {"status": "success"}
