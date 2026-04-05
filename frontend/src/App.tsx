import { BrowserRouter, Routes, Route } from "react-router-dom"
import AdminLayout from "./layouts/AdminLayout"
import Dashboard from "./pages/Dashboard"
import { Button } from "./components/ui/button"

function Placeholder({ title }: { title: string }) {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold tracking-tight">{title}</h2>
      <div className="rounded-xl border border-border bg-card text-card-foreground shadow p-12 text-center text-muted-foreground flex flex-col items-center justify-center">
        {title} 面板 - 等待接入核心数据
        <Button className="mt-6" variant="outline">执行熔断 / 同步</Button>
      </div>
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AdminLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="channels" element={<Placeholder title="千问上游池 (Channels)" />} />
          <Route path="tokens" element={<Placeholder title="下游分发权柄 (Tokens)" />} />
          <Route path="settings" element={<Placeholder title="系统级独裁设置" />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
