import { Flex, Typography } from 'antd'
import type { ReactNode } from 'react'

type PageContainerProps = {
  title: string
  description?: string
  extra?: ReactNode
  children: ReactNode
}

export const PageContainer = ({ title, description, extra, children }: PageContainerProps) => (
  <section className="page-container">
    <Flex justify="space-between" align="center" className="page-header">
      <div>
        <Typography.Title level={3}>{title}</Typography.Title>
        {description && <Typography.Text type="secondary">{description}</Typography.Text>}
      </div>
      {extra}
    </Flex>
    <div className="page-body">{children}</div>
  </section>
)
