import { AppstoreOutlined, DatabaseOutlined, DeploymentUnitOutlined, ExperimentOutlined, RocketOutlined, SettingOutlined, ThunderboltOutlined } from '@ant-design/icons'
import type { ReactNode } from 'react'

export type WorkspaceNavItem = {
  key: string
  label: string
  path: string
  description: string
  icon: ReactNode
}

export const workspaceNavItems: WorkspaceNavItem[] = [
  {
    key: 'quick-start',
    label: 'Quick Start',
    path: '/app/quick-start',
    description: 'Guided wizard for training tasks',
    icon: <RocketOutlined />,
  },
  {
    key: 'tasks',
    label: 'Tasks',
    path: '/app/tasks',
    description: 'Monitor queued and running jobs',
    icon: <AppstoreOutlined />,
  },
  {
    key: 'datasets',
    label: 'Datasets',
    path: '/app/datasets',
    description: 'Manage corpora, preview samples',
    icon: <DatabaseOutlined />,
  },
  {
    key: 'chat-lab',
    label: 'Chat Lab',
    path: '/app/chat-lab',
    description: 'Host multimodal inference consoles',
    icon: <ThunderboltOutlined />,
  },
  {
    key: 'eval-lab',
    label: 'Eval Lab',
    path: '/app/eval-lab',
    description: 'Benchmark checkpoints and prompts',
    icon: <ExperimentOutlined />,
  },
  {
    key: 'export-hub',
    label: 'Export Hub',
    path: '/app/export-hub',
    description: 'Package adapters or quantized artifacts',
    icon: <DeploymentUnitOutlined />,
  },
  {
    key: 'admin',
    label: 'Admin',
    path: '/app/admin',
    description: 'Resource quotas, policies, and audit logs',
    icon: <SettingOutlined />,
  },
]
