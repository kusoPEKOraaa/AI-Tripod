export type ModelPreset = {
  id: string
  name: string
  family: string
  params: string
  tags: string[]
}

export type TrainingTask = {
  id: string
  name: string
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  owner: string
  accelerator: string
  createdAt: string
  stage: 'sft' | 'dpo' | 'eval'
}

export type DatasetItem = {
  id: string
  name: string
  size: string
  domain: string
  stage: 'sft' | 'pretrain' | 'eval' | 'dpo'
  samples: Array<Record<string, string>>
}

const wait = (ms = 400) => new Promise((resolve) => setTimeout(resolve, ms))

export const getAvailableModels = async () => {
  await wait()
  const models: ModelPreset[] = [
    { id: 'qwen2.5-7b', name: 'Qwen2.5-7B', family: 'Qwen', params: '7B', tags: ['热门', 'LLM'] },
    { id: 'deepseek-r1', name: 'DeepSeek-R1 8B', family: 'DeepSeek', params: '8B', tags: ['推理'] },
    { id: 'llama3.1-70b', name: 'Llama3.1-70B', family: 'Meta', params: '70B', tags: ['企业', '资源大'] },
  ]
  return models
}

export const getMockTasks = async () => {
  await wait()
  const tasks: TrainingTask[] = [
    {
      id: 'task-001',
      name: 'Qwen2.5 中文客服 SFT',
      status: 'running',
      owner: 'ops-bot',
      accelerator: 'A100 * 4',
      createdAt: '2024-11-18T12:00:00Z',
      stage: 'sft',
    },
    {
      id: 'task-002',
      name: 'Llama3.1 医疗 DPO',
      status: 'queued',
      owner: 'care-team',
      accelerator: 'H20 * 8',
      createdAt: '2024-11-19T02:00:00Z',
      stage: 'dpo',
    },
    {
      id: 'task-003',
      name: 'DeepSeek-R1 推理评测',
      status: 'succeeded',
      owner: 'research',
      accelerator: '4090 * 2',
      createdAt: '2024-11-17T05:00:00Z',
      stage: 'eval',
    },
  ]
  return tasks
}

export const getMockDatasets = async () => {
  await wait()
  const datasets: DatasetItem[] = [
    {
      id: 'ds-001',
      name: '中文客服多轮对话',
      size: '45K',
      domain: '客服',
      stage: 'sft',
      samples: [
        { instruction: '您好，介绍一下业务办理流程。', output: '当然，以下是步骤...' },
        { instruction: '请提供退款申请链接。', output: '请访问 https://ai-tripod/refund' },
      ],
    },
    {
      id: 'ds-002',
      name: '医疗问诊 QA',
      size: '12K',
      domain: '医疗',
      stage: 'dpo',
      samples: [
        { question: '发烧伴随头痛怎么办？', answer: '保持补水，必要时就医。' },
      ],
    },
  ]
  return datasets
}
