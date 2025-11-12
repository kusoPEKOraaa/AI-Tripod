import { Card, Col, Progress, Row, Table, Tag } from 'antd'
import { PageContainer } from '@/components/layout/PageContainer'

const columns = [
  { title: '用户', dataIndex: 'user' },
  { title: '角色', dataIndex: 'role' },
  { title: '允许 GPU', dataIndex: 'gpus' },
  {
    title: '任务类型',
    dataIndex: 'stages',
    render: (stages: string[]) => stages.map((stage) => <Tag key={stage}>{stage}</Tag>),
  },
]

const dataSource = [
  { key: '1', user: 'ops-bot', role: '管理员', gpus: '0,1,2,3', stages: ['SFT', 'DPO', 'EVAL'] },
  { key: '2', user: 'research', role: '开发者', gpus: '4,5', stages: ['SFT', 'EVAL'] },
]

export const AdminPage = () => (
  <PageContainer
    title="算力与权限配置"
    description="管理员可按团队维度限制 GPU、任务类型与运行时参数上限。"
  >
    <Row gutter={16} style={{ marginBottom: 16 }}>
      <Col span={8}>
        <Card>
          <Progress type="dashboard" percent={68} />
          <p style={{ textAlign: 'center', marginTop: 8 }}>GPU 占用率</p>
        </Card>
      </Col>
      <Col span={8}>
        <Card>
          <Progress type="dashboard" percent={35} strokeColor="#52c41a" />
          <p style={{ textAlign: 'center', marginTop: 8 }}>本周完成任务</p>
        </Card>
      </Col>
      <Col span={8}>
        <Card>
          <Progress type="dashboard" percent={12} strokeColor="#f5222d" />
          <p style={{ textAlign: 'center', marginTop: 8 }}>告警率</p>
        </Card>
      </Col>
    </Row>
    <Card>
      <Table rowKey="key" columns={columns} dataSource={dataSource} pagination={false} />
    </Card>
  </PageContainer>
)
