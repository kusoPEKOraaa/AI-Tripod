import { Layout, Menu, Typography } from 'antd'
import { useLocation, useNavigate } from 'react-router-dom'
import { workspaceNavItems } from '@/constants/navigation'

const { Sider } = Layout

export const AppSidebar = () => {
  const location = useLocation()
  const navigate = useNavigate()

  const selected = workspaceNavItems.find((item) => location.pathname.startsWith(item.path))

  return (
    <Sider width={240} className="workspace-sidebar">
      <div className="sidebar-brand">
        <Typography.Title level={4}>AI Tripod</Typography.Title>
        <Typography.Text type="secondary">LLaMA-Factory Inspired</Typography.Text>
      </div>
      <Menu
        mode="inline"
        selectedKeys={selected ? [selected.key] : []}
        items={workspaceNavItems.map((item) => ({
          key: item.key,
          icon: item.icon,
          label: item.label,
          onClick: () => navigate(item.path),
        }))}
      />
    </Sider>
  )
}
