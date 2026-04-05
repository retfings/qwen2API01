import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import settings
from backend.core.database import AsyncJsonDB
from backend.core.browser_engine import BrowserEngine
from backend.core.account_pool import AccountPool
from backend.services.qwen_client import QwenClient
from backend.api import admin, v1_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("qwen2api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting qwen2API v2.0 Enterprise Gateway...")
    
    # 初始化数据存储 (带锁 JSON)
    app.state.accounts_db = AsyncJsonDB(settings.ACCOUNTS_FILE, default_data=[])
    app.state.users_db = AsyncJsonDB(settings.USERS_FILE, default_data=[])
    
    # 初始化组件
    app.state.browser_engine = BrowserEngine(pool_size=settings.BROWSER_POOL_SIZE)
    app.state.account_pool = AccountPool(app.state.accounts_db, max_inflight=settings.MAX_INFLIGHT_PER_ACCOUNT)
    app.state.qwen_client = QwenClient(app.state.browser_engine, app.state.account_pool)
    
    # 启动引擎与加载账号
    await app.state.account_pool.load()
    asyncio.create_task(app.state.browser_engine.start())
    
    yield
    
    log.info("Shutting down gateway...")
    await app.state.browser_engine.stop()

app = FastAPI(title="qwen2API Enterprise Gateway", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载路由
app.include_router(v1_chat.router, prefix="/v1/chat", tags=["OpenAI Compatible"])
app.include_router(admin.router, prefix="/api/admin", tags=["Dashboard Admin"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=settings.PORT, workers=1)
