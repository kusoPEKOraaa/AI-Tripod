import { Card, Statistic, Typography } from 'antd'

type StatCardProps = {
  label: string
  value: number | string
  suffix?: string
  trend?: string
}

export const StatCard = ({ label, value, suffix, trend }: StatCardProps) => (
  <Card bordered={false} className="stat-card">
    <Statistic title={label} value={value} suffix={suffix} />
    {trend && (
      <Typography.Text type="secondary" style={{ marginTop: 8, display: 'block' }}>
        {trend}
      </Typography.Text>
    )}
  </Card>
)
