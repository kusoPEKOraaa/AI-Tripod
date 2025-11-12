import { useNavigate } from 'react-router-dom'
import { Button, Card, Col, Flex, Row, Space, Statistic, Tag, Timeline, Typography } from 'antd'
import { FireOutlined, ThunderboltOutlined, SafetyOutlined, ApiOutlined, CloudServerOutlined, RobotOutlined, SyncOutlined } from '@ant-design/icons'

const featureCards = [
  {
    title: '极速训练编排',
    description: '图形化向导与 JSON 参数预览并行，串联 LLaMA-Factory 的数据-训练-评测-导出流水线。',
    icon: <ThunderboltOutlined />,
  },
  {
    title: '集群级资源守护',
    description: '管理员限制可用 GPU、任务类型与并发度，结合 Django 权限体系实现多租户隔离。',
    icon: <CloudServerOutlined />,
  },
  {
    title: '多模态推理工作台',
    description: 'Chat Lab 支持图片/视频/音频与工具调用，实现与原型一致的多模态体验。',
    icon: <SafetyOutlined />,
  },
  {
    title: '一键导出/部署',
    description: 'Export Hub 兼容 GGUF、GPTQ、LoRA Adapter，提供推送至模型仓库的自动流程。',
    icon: <SyncOutlined />,
  },
]

const promoModels = ['Qwen3-8B', 'DeepSeek-R1-8B', 'Llama3.1-70B', 'InternLM2', 'Yi-1.5']

const pipelineSteps = [
  { title: '模型配置', description: '支持 100+ 基座模型、量化与 Booster，保留 LLaMA-Factory 的 Config 行为。' },
  { title: '数据管理', description: '数据集集中托管，支持元信息校验、样本预览与权限控制。' },
  { title: '训练调度', description: '与 runner API 对接，可查看日志、Loss 曲线、Deepspeed 配置。' },
  { title: '评测&推理', description: 'Eval Lab & Chat Lab 还原原型能力，便于快速验证效果。' },
  { title: '导出&上线', description: 'GGUF / GPTQ / Adapter 导出，提供 CLI 与在线命令同步。' },
]

const stats = [
  { title: '模型与数据集', value: 120, suffix: '+', description: '官方模板与企业专用资源' },
  { title: 'GPU 节点', value: 48, suffix: '台', description: '可按租户拆分的 A100/H20 集群' },
  { title: '成功作业', value: 986, suffix: '', description: '已有历史任务可复现重跑' },
]

export const LandingPage = () => {
  const navigate = useNavigate()

  return (
    <div className="landing-page">
      <nav className="landing-nav">
        <Typography.Title level={4} style={{ margin: 0 }}>
          AI Tripod
        </Typography.Title>
        <Space size="large">
          <a href="#capabilities">产品能力</a>
          <a href="#pipeline">流程</a>
          <a href="#workspace">控制台</a>
          <a href="#cta">立即体验</a>
        </Space>
        <Button type="primary" size="large" onClick={() => navigate('/app/quick-start')}>
          进入控制台
        </Button>
      </nav>

      <header className="landing-hero">
        <Tag color="gold" icon={<FireOutlined />}>
          LoRA 模型对话限时免费
        </Tag>
        <Typography.Title>大模型集成训练平台</Typography.Title>
        <Typography.Paragraph>
          复刻 LLaMA-Factory WebUI 的交互体验，结合 Django + React 构建企业级多模态训练、评测、推理与导出工作台。
        </Typography.Paragraph>
        <Space size="middle" wrap>
          <Button type="primary" size="large" onClick={() => navigate('/app/quick-start')}>
            立即启动训练
          </Button>
          <Button size="large" onClick={() => navigate('/app/tasks')}>
            查看任务看板
          </Button>
        </Space>
        <div className="landing-hero-stats">
          {stats.map((item) => (
            <Card key={item.title} bordered={false}>
              <Statistic title={item.title} value={item.value} suffix={item.suffix} />
              <Typography.Text type="secondary">{item.description}</Typography.Text>
            </Card>
          ))}
        </div>
      </header>

      <section className="landing-models" id="capabilities">
        <Typography.Title level={4}>精选基础模型</Typography.Title>
        <Space wrap>
          {promoModels.map((name) => (
            <Tag key={name} color="purple">
              {name}
            </Tag>
          ))}
        </Space>
      </section>

      <section className="landing-features">
        <Row gutter={[24, 24]}>
          {featureCards.map((card) => (
            <Col key={card.title} xs={24} md={12} lg={6}>
              <Card hoverable className="feature-card">
                <Flex vertical gap={8}>
                  <div className="feature-icon">{card.icon}</div>
                  <Typography.Title level={5}>{card.title}</Typography.Title>
                  <Typography.Text type="secondary">{card.description}</Typography.Text>
                </Flex>
              </Card>
            </Col>
          ))}
        </Row>
      </section>

      <section className="landing-pipeline" id="pipeline">
        <Typography.Title level={4}>与 LLaMA-Factory 一致的研发流程</Typography.Title>
        <Timeline
          mode="alternate"
          items={pipelineSteps.map((step) => ({
            dot: <RobotOutlined />,
            children: (
              <div>
                <Typography.Title level={5}>{step.title}</Typography.Title>
                <Typography.Text type="secondary">{step.description}</Typography.Text>
              </div>
            ),
          }))}
        />
      </section>

      <section className="landing-console" id="workspace">
        <Card className="console-card" bodyStyle={{ padding: 0 }}>
          <div className="console-preview">
            <div className="console-preview__title">
              <div className="dot red" />
              <div className="dot yellow" />
              <div className="dot green" />
              <Typography.Text>Workspace Preview</Typography.Text>
            </div>
            <div className="console-preview__content">
              <div className="sidebar-placeholder">
                <ApiOutlined />
                <Typography.Text>导航</Typography.Text>
              </div>
              <div className="content-placeholder">
                <Typography.Title level={5}>Quick Start</Typography.Title>
                <Typography.Paragraph type="secondary">
                  四步向导、CLI 预览、实时校验以及作业看板一次到位，UI 布局已与题目要求对齐。
                </Typography.Paragraph>
              </div>
            </div>
          </div>
        </Card>
      </section>

      <section className="landing-cta" id="cta">
        <Typography.Title level={3}>准备好复刻 LLaMA-Factory 在线平台了吗？</Typography.Title>
        <Typography.Paragraph>
          立即进入 React + Django 工作台，体验训练、评测、推理、导出的一体化流程。
        </Typography.Paragraph>
        <Space size="large" wrap>
          <Button type="primary" size="large" onClick={() => navigate('/app/quick-start')}>
            创建任务
          </Button>
          <Button size="large" type="default" onClick={() => navigate('/app/chat-lab')}>
            体验 Chat Lab
          </Button>
        </Space>
      </section>
    </div>
  )
}
