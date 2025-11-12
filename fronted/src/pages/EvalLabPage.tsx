import { Button, Card, Col, Form, InputNumber, Row, Select } from 'antd'
import { PageContainer } from '@/components/layout/PageContainer'
import { EmptyState } from '@/components/feedback/EmptyState'

export const EvalLabPage = () => (
  <PageContainer
    title="Evaluate & Predict"
    description="构建快速评测、批量预测任务，与 LLaMA-Factory Eval 标签对齐。"
    extra={<Button type="primary">启动评测</Button>}
  >
    <Row gutter={16}>
      <Col span={12}>
        <Card title="评测配置" bordered={false}>
          <Form layout="vertical" initialValues={{ dataset: 'alpaca_eval', temperature: 0.7 }}>
            <Form.Item label="输入数据集" name="dataset">
              <Select
                options={[
                  { label: 'Alpaca Eval', value: 'alpaca_eval' },
                  { label: 'MT-Bench', value: 'mt_bench' },
                ]}
              />
            </Form.Item>
            <Form.Item label="max_new_tokens" name="maxNewTokens">
              <InputNumber min={8} max={2048} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item label="temperature" name="temperature">
              <InputNumber min={0} max={1.5} step={0.05} style={{ width: '100%' }} />
            </Form.Item>
          </Form>
        </Card>
      </Col>
      <Col span={12}>
        <Card title="评测结果" bordered={false}>
          <EmptyState title="暂无评测任务" description="创建任务后将展示分数与日志。" />
        </Card>
      </Col>
    </Row>
  </PageContainer>
)
