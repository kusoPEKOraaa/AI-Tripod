import { Layout } from 'antd'
import { Outlet } from 'react-router-dom'
import { AppSidebar } from '@/components/navigation/AppSidebar'
import { AppTopbar } from '@/components/navigation/AppTopbar'

const { Content } = Layout

export const WorkspaceLayout = () => (
  <Layout className="workspace-shell">
    <AppSidebar />
    <Layout className="workspace-main">
      <AppTopbar />
      <Content className="workspace-content">
        <Outlet />
      </Content>
    </Layout>
  </Layout>
)
