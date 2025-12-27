import { useMemo } from 'react'
import { Button, Card, List, Space, Tabs, Tag, Typography } from 'antd'
import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { PageContainer } from '@/components/layout/PageContainer'
import { StatCard } from '@/components/cards/StatCard'
import { getMockTasks } from '@/services/mockApi'
import type { TrainingTask } from '@/services/mockApi'

const statusColor: Record<TrainingTask['status'], string> = {
  queued: 'default',
  running: 'processing',
  succeeded: 'success',
  failed: 'error',
}

export const TasksPage = () => {
  const { data: tasks } = useQuery({ queryKey: ['tasks'], queryFn: getMockTasks })
  const navigate = useNavigate()

  const grouped = useMemo(() => {
    const output: Record<TrainingTask['status'], TrainingTask[]> = {
      queued: [],
      running: [],
      succeeded: [],
      failed: [],
    }
    tasks?.forEach((task) => output[task.status].push(task))
    return output
  }, [tasks])

  const summary = [
    { label: '排队中', value: grouped?.queued?.length ?? 0 },
    { label: '运行中', value: grouped?.running?.length ?? 0 },
    { label: '已完成', value: grouped?.succeeded?.length ?? 0 },
  ]

  return (
    <PageContainer
      title="任务监控"
      description="来自 LLaMA-Factory runner 的实时作业状态、日志与指标。"
      extra={
        <Space>
          <Button onClick={() => navigate('/app/quick-start')}>新建任务</Button>
          <Button type="primary">刷新数据</Button>
        </Space>
      }
    >
      <Space size="large" style={{ width: '100%', marginBottom: 16 }} wrap>
        {summary.map((card) => (
          <StatCard key={card.label} label={card.label} value={card.value} />
        ))}
      </Space>

      <Card>
        <Tabs
          defaultActiveKey="running"
          items={['running', 'queued', 'succeeded', 'failed'].map((status) => ({
            key: status,
            label: `${status.toUpperCase()} (${grouped?.[status as keyof typeof grouped]?.length ?? 0})`,
            children: (
              <List
                dataSource={grouped?.[status as keyof typeof grouped] ?? []}
                renderItem={(item) => (
                  <List.Item
                    actions={[
                      <Button size="small" key="detail">
                        查看日志
                      </Button>,
                    ]}
                  >
                    <List.Item.Meta
                      title={
                        <Space>
                          <Typography.Text strong>{item.name}</Typography.Text>
                          <Tag color={statusColor[item.status]}>{item.status}</Tag>
                        </Space>
                      }
                      description={
                        <Space split={<span>•</span>}>
                          <span>账号：{item.owner}</span>
                          <span>算力：{item.accelerator}</span>
                          <span>创建时间：{new Date(item.createdAt).toLocaleString()}</span>
                        </Space>
                      }
                    />
                  </List.Item>
                )}
              />
            ),
          }))}
        />
      </Card>
    </PageContainer>
  )
}
