import { Alert, Button, Card, Flex, Input, Space, Tabs } from 'antd'
import { EmptyState } from '@/components/feedback/EmptyState'
import { PageContainer } from '@/components/layout/PageContainer'

export const ChatLabPage = () => (
  <PageContainer
    title="Chat / 推理工作台"
    description="复刻 LLaMA-Factory Chat 标签，提供多模态消息、工具调用与 streaming。"
    extra={<Button type="primary">装载模型</Button>}
  >
    <Alert
      type="info"
      showIcon
      message="当前为前端骨架，待后端 API 接入后可实时加载推理引擎。"
      style={{ marginBottom: 16 }}
    />
    <Card>
      <Flex gap={16} wrap>
        <div className="chat-pane">
          <Tabs
            items={[
              { key: 'chat', label: '对话记录', children: <EmptyState title="暂未加载模型" description="点击右上角按钮装载模型后开始对话。" /> },
              { key: 'tools', label: '函数调用', children: <Input.TextArea rows={6} placeholder="JSON schema" /> },
            ]}
          />
        </div>
        <div className="chat-controls">
          <Space direction="vertical" style={{ width: '100%' }}>
            <Input.TextArea placeholder="请输入 Prompt" rows={8} />
            <Button type="primary" block>
              发送
            </Button>
          </Space>
        </div>
      </Flex>
    </Card>
  </PageContainer>
)
