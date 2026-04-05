import { useEffect, useState } from "react"
import { Users, Server, Activity, ShieldAlert, ActivityIcon } from "lucide-react"

export default function Dashboard() {
  const [status, setStatus] = useState<any>(null)

  useEffect(() => {
    fetch("/api/admin/status", { headers: { Authorization: "Bearer admin" } })
      .then(res => res.json())
      .then(data => setStatus(data))
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">API 分发中枢 (Dashboard)</h2>
        <p className="text-muted-foreground">全局并发监控与千问账号池概览。</p>
      </div>
      
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border bg-card text-card-foreground shadow p-6">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="tracking-tight text-sm font-medium">可用上游账号</h3>
            <Server className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="text-2xl font-bold">{status?.accounts?.valid || 0}</div>
        </div>

        <div className="rounded-xl border bg-card text-card-foreground shadow p-6">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="tracking-tight text-sm font-medium">当前并发引擎</h3>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="text-2xl font-bold">{status?.browser_engine?.pool_size || 0} Pages</div>
        </div>

        <div className="rounded-xl border bg-card text-card-foreground shadow p-6">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="tracking-tight text-sm font-medium text-destructive">排队请求数</h3>
            <ShieldAlert className="h-4 w-4 text-destructive" />
          </div>
          <div className="text-2xl font-bold text-destructive">{status?.browser_engine?.queue || 0}</div>
        </div>

        <div className="rounded-xl border bg-card text-card-foreground shadow p-6">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="tracking-tight text-sm font-medium">限流号/死号</h3>
            <ActivityIcon className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="text-2xl font-bold">{status?.accounts?.rate_limited || 0} / {status?.accounts?.invalid || 0}</div>
        </div>
      </div>
    </div>
  )
}
