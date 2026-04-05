import asyncio
import logging
import time
from typing import Optional
from backend.core.database import AsyncJsonDB
from backend.core.config import settings

log = logging.getLogger("qwen2api.accounts")

class Account:
    def __init__(self, email="", password="", token="", cookies="", username="", activation_pending=False, **kwargs):
        self.email = email
        self.password = password
        self.token = token
        self.cookies = cookies
        self.username = username
        self.activation_pending = activation_pending
        self.valid = not activation_pending
        self.last_used = 0.0
        self.inflight = 0
        self.rate_limited_until = 0.0

    def is_rate_limited(self) -> bool:
        return self.rate_limited_until > time.time()

    def is_available(self) -> bool:
        return self.valid and not self.is_rate_limited()

    def to_dict(self):
        return {
            "email": self.email,
            "password": self.password,
            "token": self.token,
            "cookies": self.cookies,
            "username": self.username,
            "activation_pending": self.activation_pending
        }

class AccountPool:
    def __init__(self, db: AsyncJsonDB, max_inflight: int = settings.MAX_INFLIGHT_PER_ACCOUNT):
        self.db = db
        self.max_inflight = max_inflight
        self.accounts: list[Account] = []
        self._lock = asyncio.Lock()
        self._waiters: list[asyncio.Event] = []
        self._sticky_email: Optional[str] = None

    async def load(self):
        data = await self.db.load()
        self.accounts = [Account(**d) for d in data] if isinstance(data, list) else []
        log.info(f"Loaded {len(self.accounts)} upstream account(s)")

    async def save(self):
        await self.db.save([a.to_dict() for a in self.accounts])

    async def add(self, account: Account):
        async with self._lock:
            self.accounts = [a for a in self.accounts if a.email != account.email]
            self.accounts.append(account)
        await self.save()

    async def remove(self, email: str):
        async with self._lock:
            self.accounts = [a for a in self.accounts if a.email != email]
        await self.save()

    async def acquire(self, exclude: set = None) -> Optional[Account]:
        async with self._lock:
            available = [a for a in self.accounts if a.is_available() and (not exclude or a.email not in exclude)]
            if not available:
                return None
            
            if self._sticky_email:
                sticky = next((a for a in available if a.email == self._sticky_email), None)
                if sticky and sticky.inflight < self.max_inflight:
                    sticky.inflight += 1
                    sticky.last_used = time.time()
                    return sticky
            
            available.sort(key=lambda a: a.inflight)
            best = available[0]
            if best.inflight >= self.max_inflight:
                return None
                
            best.inflight += 1
            best.last_used = time.time()
            self._sticky_email = best.email
            return best

    async def acquire_wait(self, timeout: float = 60, exclude: set = None) -> Optional[Account]:
        acc = await self.acquire(exclude)
        if acc: return acc
        
        evt = asyncio.Event()
        self._waiters.append(evt)
        try:
            await asyncio.wait_for(evt.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            if evt in self._waiters:
                self._waiters.remove(evt)
        return await self.acquire(exclude)

    def release(self, acc: Account):
        acc.inflight = max(0, acc.inflight - 1)
        if self._waiters:
            evt = self._waiters.pop(0)
            evt.set()

    def mark_invalid(self, acc: Account):
        acc.valid = False
        if self._sticky_email == acc.email:
            self._sticky_email = None
        log.warning(f"[AccountPool] {acc.email} marked invalid. Circuit broken.")

    def mark_rate_limited(self, acc: Account, cooldown: int = settings.RATE_LIMIT_COOLDOWN):
        acc.rate_limited_until = time.time() + cooldown
        if self._sticky_email == acc.email:
            self._sticky_email = None
        log.warning(f"[AccountPool] {acc.email} rate limited for {cooldown}s.")

    def status(self):
        available = [a for a in self.accounts if a.is_available()]
        rate_limited = [a for a in self.accounts if a.valid and a.is_rate_limited()]
        invalid = [a for a in self.accounts if not a.valid]
        in_use = sum(a.inflight for a in available)
        return {
            "total": len(self.accounts),
            "valid": len(available),
            "rate_limited": len(rate_limited),
            "invalid": len(invalid),
            "in_use": in_use,
            "max_inflight": self.max_inflight,
            "waiting": len(self._waiters),
        }
