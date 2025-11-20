import { useEffect, useState } from 'react'
import { Button, Card, Col, Collapse, Form, InputNumber, Row, Select, Slider, Space, Steps, message } from 'antd'
import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { PageContainer } from '@/components/layout/PageContainer'
import { getAvailableModels } from '@/services/mockApi'
import { useGlobalConfigStore } from '@/stores/useGlobalConfig'

const datasetOptions = [
  { label: '中文客服多轮对话', value: 'support_multi_turn' },
  { label: '医疗问诊 QA', value: 'medical_qa' },
  { label: '公开指令数据（混合）', value: 'open_mix' },
]

export const QuickStartPage = () => {
  const navigate = useNavigate()
  const [current, setCurrent] = useState(0)
  const [form] = Form.useForm()
  const previewValues = Form.useWatch([], form)
  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: getAvailableModels,
  })
  const hub = useGlobalConfigStore((state) => state.hub)
  const finetuningType = useGlobalConfigStore((state) => state.finetuningType)
  const updateConfig = useGlobalConfigStore((state) => state.updateConfig)

  const onSubmit = async () => {
    const values = await form.validateFields()
    updateConfig({
      selectedModel: values.model,
      hub: values.hub,
      finetuningType: values.finetuningType,
    })
    message.success('任务已加入调度队列')
    navigate('/app/tasks')
  }

  const wizardSteps = [
    { title: '模型配置', description: '选择基础模型与微调方式' },
    { title: '数据集', description: '绑定训练/验证数据' },
    { title: '超参数', description: '控制 batch、lr、scheduler' },
    { title: '资源', description: '指定算力、监控参数' },
  ]

  const modelOptions = (models ?? []).map((model) => ({
    label: `${model.name} · ${model.params}`,
    value: model.id,
  }))

  // 使用 useEffect 同步初始值，避免无限循环
  useEffect(() => {
    form.setFieldsValue({
      hub,
      finetuningType,
      learningRate: 5e-5,
      batchSize: 2,
      maxSteps: 500,
      gpuCount: 2,
      dataset: 'support_multi_turn',
    })
  }, [form, hub, finetuningType])

  return (
    <PageContainer
      title="训练任务快速启动"
      description="按照 LLaMA-Factory 的参数体系组织的 4 步向导，生成 CLI 与配置文件。"
      extra={
        <Space>
          <Button onClick={() => form.resetFields()}>重置</Button>
          <Button type="primary" onClick={onSubmit}>
            启动作业
          </Button>
        </Space>
      }
    >
      <Card>
        <Steps
          current={current}
          items={wizardSteps.map((step) => ({ key: step.title, title: step.title, description: step.description }))}
          onChange={setCurrent}
        />
        <Form layout="vertical" form={form} style={{ marginTop: 24 }}>
          {current === 0 && (
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="基础模型" name="model" rules={[{ required: true, message: '请选择模型' }]}>
                  <Select options={modelOptions} placeholder="请选择已适配模型" showSearch />
                </Form.Item>
              </Col>
              <Col span={6}>
                <Form.Item label="模型来源" name="hub">
                  <Select
                    options={[
                      { label: 'HuggingFace', value: 'huggingface' },
                      { label: 'ModelScope', value: 'modelscope' },
                      { label: 'OpenMind', value: 'openmind' },
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col span={6}>
                <Form.Item label="微调方式" name="finetuningType">
                  <Select
                    options={[
                      { label: 'LoRA', value: 'lora' },
                      { label: 'QLoRA', value: 'qlora' },
                      { label: 'Full Fine-tune', value: 'full' },
                    ]}
                  />
                </Form.Item>
              </Col>
            </Row>
          )}

          {current === 1 && (
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="训练数据集" name="dataset" rules={[{ required: true }]}>
                  <Select options={datasetOptions} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="验证集切分" name="valSize">
                  <Slider min={0} max={0.3} step={0.05} tooltip={{ formatter: (val) => `${(val ?? 0) * 100}%` }} />
                </Form.Item>
              </Col>
            </Row>
          )}

          {current === 2 && (
            <Row gutter={16}>
              <Col span={8}>
                <Form.Item label="Learning Rate" name="learningRate">
                  <InputNumber addonAfter="lr" style={{ width: '100%' }} step={0.00001} min={0.000001} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label="Batch Size" name="batchSize">
                  <InputNumber min={1} max={64} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label="最大 Step 数" name="maxSteps">
                  <InputNumber min={100} max={5000} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
          )}

          {current === 3 && (
            <Row gutter={16}>
              <Col span={8}>
                <Form.Item label="GPU 数量" name="gpuCount">
                  <InputNumber min={1} max={16} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label="DeepSpeed 阶段" name="dsStage">
                  <Select
                    options={[
                      { label: '关闭', value: 'none' },
                      { label: 'Stage 2', value: '2' },
                      { label: 'Stage 3', value: '3' },
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label="监控上报" name="telemetry">
                  <Select
                    options={[
                      { label: '关闭', value: 'none' },
                      { label: 'W&B', value: 'wandb' },
                      { label: 'SwanLab', value: 'swanlab' },
                    ]}
                  />
                </Form.Item>
              </Col>
            </Row>
          )}
        </Form>
      </Card>

      <Collapse
        style={{ marginTop: 24 }}
        items={[
          {
            key: 'preview',
            label: 'CLI 参数预览',
            children: <CodePreview values={previewValues ?? form.getFieldsValue(true)} />,
          },
        ]}
      />
    </PageContainer>
  )
}

type CodePreviewProps = {
  values: Record<string, unknown>
}

const CodePreview = ({ values }: CodePreviewProps) => {
  const safe = (field: string, fallback = ''): string => {
    const raw = values?.[field]
    if (typeof raw === 'string' || typeof raw === 'number' || typeof raw === 'boolean') {
      return String(raw)
    }
    return fallback
  }

  return (
    <pre className="code-preview">
      llamafactory-cli train \
      {'\n'} --model {safe('model', '<model>')} \
      {'\n'} --hub {safe('hub', 'huggingface')} \
      {'\n'} --dataset {safe('dataset', 'dataset')} \
      {'\n'} --batch_size {safe('batchSize', '2')} \
      {'\n'} --learning_rate {safe('learningRate', '5e-5')}
    </pre>
  )
}
