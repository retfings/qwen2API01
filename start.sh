#!/bin/bash

# ==========================================
# qwen2API Enterprise Gateway - 控制中枢点火脚本
# ==========================================

echo "⚡ 正在唤醒底层铁壁 (Backend)..."
cd backend || exit 1
# 如果没有虚拟环境，自动创建或提示，这里假设直接用全局或现成环境
nohup python main.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "✓ Backend 已点火 (PID: $BACKEND_PID) -> 日志: logs/backend.log"
cd ..

echo "⚡ 正在唤醒前端王座 (Admin Dashboard)..."
cd frontend || exit 1
nohup npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "✓ Frontend 已点火 (PID: $FRONTEND_PID) -> 日志: logs/frontend.log"
cd ..

echo ""
echo "=========================================="
echo "帝国已上线。"
echo "▶ 前端中枢: http://localhost:5173"
echo "▶ 后端核心: http://localhost:8080"
echo "=========================================="
echo "如需掐断进程，请运行: kill $BACKEND_PID $FRONTEND_PID"
