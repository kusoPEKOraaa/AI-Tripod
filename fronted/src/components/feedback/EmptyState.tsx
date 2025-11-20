import { Empty, Typography } from 'antd'
import type { ReactNode } from 'react'

type EmptyStateProps = {
  title: string
  description?: string
  actions?: ReactNode
}

export const EmptyState = ({ title, description, actions }: EmptyStateProps) => (
  <div className="empty-state">
    <Empty description={false} />
    <Typography.Title level={4}>{title}</Typography.Title>
    {description && <Typography.Text type="secondary">{description}</Typography.Text>}
    {actions}
  </div>
)
