import { Button, Card, Col, Form, Input, InputNumber, Row, Select, Switch } from 'antd'
import { PageContainer } from '@/components/layout/PageContainer'

export const ExportHubPage = () => (
  <PageContainer
    title="导出中心"
    description="同步 LLaMA-Factory Export 标签，可生成 GGUF/GPTQ/LoRA 产物。"
    extra={<Button type="primary">执行导出</Button>}
  >
    <Card>
      <Form layout="vertical">
        <Row gutter={16}>
          <Col span={8}>
            <Form.Item label="导出格式" name="format">
              <Select
                options={[
                  { value: 'gguf', label: 'GGUF' },
                  { value: 'gptq', label: 'GPTQ' },
                  { value: 'safetensors', label: 'SafeTensors' },
                ]}
              />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item label="量化比特">
              <InputNumber min={2} max={16} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item label="输出目录">
              <Input placeholder="saves/qwen2.5-lora/export" />
            </Form.Item>
          </Col>
        </Row>
        <Row gutter={16}>
          <Col span={8}>
            <Form.Item label="导出分片数">
              <InputNumber min={1} max={20} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item label="推送至仓库">
              <Input placeholder="org/project-export" />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item label="启用校验">
              <Switch />
            </Form.Item>
          </Col>
        </Row>
      </Form>
    </Card>
  </PageContainer>
)
