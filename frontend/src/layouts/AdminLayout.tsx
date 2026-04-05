import { Outlet, Link, useLocation } from "react-router-dom"
import { Activity, Key, Settings, LayoutDashboard } from "lucide-react"

export default function AdminLayout() {
  const loc = useLocation()
  
  const navs = [
    { name: "大盘监控", path: "/", icon: LayoutDashboard },
    { name: "千问池 (Channels)", path: "/channels", icon: Activity },
    { name: "下游分发 (Tokens)", path: "/tokens", icon: Key },
    { name: "系统设置", path: "/settings", icon: Settings },
  ]

  return (
    <div className="flex min-h-screen w-full bg-background">
      <aside className="w-64 flex-col border-r bg-card flex hidden md:flex">
        <div className="h-16 flex items-center px-6 border-b">
          <div className="font-bold text-lg tracking-tight text-primary">qwen2API Enterprise</div>
        </div>
        <nav className="flex-1 space-y-1 p-4">
          {navs.map(n => {
            const active = loc.pathname === n.path
            return (
              <Link 
                key={n.path}
                to={n.path} 
                className={`flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors ${
                  active ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-secondary hover:text-secondary-foreground"
                }`}
              >
                <n.icon className="h-4 w-4" />
                {n.name}
              </Link>
            )
          })}
        </nav>
      </aside>

      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="h-16 flex items-center justify-between px-6 border-b bg-card md:hidden">
           <div className="font-bold text-lg text-primary">qwen2API</div>
        </header>
        <div className="flex-1 p-6 md:p-8 overflow-y-auto">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
